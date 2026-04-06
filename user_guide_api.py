from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from gemini_client import generate_gemini
import os
import re
import traceback

# -----------------------------
# 1. App Initialization
# -----------------------------
app = Flask(__name__)
CORS(app)

# -----------------------------
# 2. Role-Based Configuration
# -----------------------------
BASE_DIR = os.getcwd()
DB_PATH = "RagDB"
ALLOWED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}

ROLE_CONFIG = {
    "Admin": {
        "collection_name": "admin_user_guide",
        "images_folder": os.path.join(BASE_DIR, "images", "admin_img"),
        "default_role": "Admin"
    },
    "Teachers": {
        "collection_name": "teacher_user_guide",
        "images_folder": os.path.join(BASE_DIR, "images", "teacher_img"),
        "default_role": "Teachers"
    },
    "Super Admin": {
        "collection_name": "superadmin_user_guide",
        "images_folder": os.path.join(BASE_DIR, "images", "super_img"),
        "default_role": "Super Admin"
    }
}

# Create all image folders
for role_data in ROLE_CONFIG.values():
    os.makedirs(role_data["images_folder"], exist_ok=True)

# -----------------------------
# 3. ChromaDB + Embedding Model
# -----------------------------
chroma_client = chromadb.PersistentClient(
    path=DB_PATH,
    settings=Settings(allow_reset=False)
)

# Pre-load all collections
collections = {}
for role, config in ROLE_CONFIG.items():
    collections[role] = chroma_client.get_collection(config["collection_name"])

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------
# 4. Utility Functions
# -----------------------------
def extract_step_number(query: str):
    match = re.search(r"\bstep\s*(\d+)\b", query.lower())
    return int(match.group(1)) if match else None


def retrieve_chunks(query: str, role: str, collection, top_k: int = 3):
    step_no = extract_step_number(query)

    # CASE 1: explicit step
    if step_no is not None:
        res = collection.get(
            where={"$and": [{"role": role}, {"order": step_no}]},
            include=["documents", "metadatas"]
        )
        if not res["documents"]:
            return []

        chunks = []
        for doc, meta in zip(res["documents"], res["metadatas"]):
            chunks.append({
                "text": doc,
                "meta": meta,
                "score": 1.0,
                "h1": meta.get("topic", ""),
                "h2": meta.get("subtopic", ""),
                "order": meta.get("order", 0),
                "part": meta.get("part", 1),
            })
        chunks.sort(key=lambda x: x["part"])
        return chunks

    # CASE 2: semantic search
    q_emb = embedding_model.encode([query])[0].tolist()
    res = collection.query(
        query_embeddings=[q_emb],
        n_results=top_k * 6,
        where={"role": role},
        include=["documents", "metadatas", "distances"]
    )

    chunks = []
    q_lower = query.lower()
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        base_score = 1.0 - dist
        h1, h2 = meta.get("topic", ""), meta.get("subtopic", "")
        bonus = 0.0
        if "student" in q_lower and "student" in (h1 + h2).lower(): 
            bonus += 0.2
        if "teacher" in q_lower and "teacher" in (h1 + h2).lower(): 
            bonus += 0.2
        chunks.append({
            "text": doc,
            "meta": meta,
            "score": round(max(0.0, base_score + bonus), 3),
            "h1": h1,
            "h2": h2,
            "order": meta.get("order", 0),
            "part": meta.get("part", 1),
        })
    chunks.sort(key=lambda x: (-x["score"], x["order"], x["part"]))
    return chunks[:top_k]


def build_prompt(query: str, chunks: list, role: str):
    context_blocks = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk["meta"]
        header = f"[{i}] {chunk['h1']}"
        if chunk["h2"]: 
            header += f" → {chunk['h2']}"
        header += f" | StepOrder={meta.get('order')} Part={meta.get('part')}"
        context_blocks.append(f"{header}\n{chunk['text']}")
    context_str = "\n\n".join(context_blocks)

    return f"""
    
You are a {role} user-guide assistant.
Use ONLY the retrieved context. Ignore unrelated sections. Always cite the section(s) used (e.g., [1], [2]).
If an image appears as [IMG:filename.ext], include it at the correct step.
Only create numbered steps if the context explicitly lists them as Step 1, Step 2, etc.
Do not convert descriptive summaries into steps.
Do not invent actions, parameters, workflows, or examples.
If no explicit steps exist, respond with a concise descriptive paragraph summarizing the relevant content from the context.
Ignore all information in the context that is unrelated to the user's query.
Keep the answer direct and focused on the question.

CONTEXT:
{context_str}

User question: {query}

Write a short, step-by-step answer.

"""


def embed_images_in_html(text, images_folder,role):
    base_url = request.host_url.rstrip("/")
    
    # Preload all available images (case-insensitive, strip spaces)
    available_images = {f.lower().replace(" ", ""): f for f in os.listdir(images_folder)}
    
    # Pattern to match [IMG:filename.ext] or [Image:filename.ext]
    pattern = re.compile(r'\[(?:Image|IMG):\s*([^\]]+?)\]', re.IGNORECASE)

    def replacer(match):
        img_name = match.group(1).strip()
        img_key = img_name.lower().replace(" ", "")
        if img_key in available_images:
            f_real = available_images[img_key]
            # Include role in the image URL paths
            return f'<img src="{base_url}/images/{role}/{f_real}" alt="{f_real}" style="max-width:800px;">'
        else:
            return f'Image not found: {img_name}'

    return pattern.sub(replacer, text)


@app.route("/images/<role>/<path:filename>", methods=["GET"])
def serve_image(role, filename):
    if role not in ROLE_CONFIG:
        return jsonify({"success": False, "error": f"Invalid role: {role}"}), 400
    
    images_folder = ROLE_CONFIG[role]["images_folder"]
    requested_lower = filename.lower()
    
    try:
        for f in os.listdir(images_folder):
            if f.lower() == requested_lower:
                return send_from_directory(images_folder, f)
        return jsonify({"success": False, "error": f"Image not found: {filename}"}), 404
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# -----------------------------
# 5. Unified RAG Query Endpoint
# -----------------------------
@app.route("/query", methods=["POST"])
def unified_query():
    try:
        data = request.get_json()
        
        # Validate request
        if not data or "query" not in data:
            return jsonify({"success": False, "error": "Missing 'query' field"}), 400
        
        if "role" not in data:
            return jsonify({"success": False, "error": "Missing 'role' field"}), 400
        
        # Extract parameters
        query = data["query"]
        role = data["role"]
        top_k = int(data.get("top_k", 3))
        top_k = max(1, min(top_k, 10))
        
        # Validate role
        if role not in ROLE_CONFIG:
            return jsonify({
                "success": False, 
                "error": f"Invalid role '{role}'. Allowed roles: {', '.join(ROLE_CONFIG.keys())}"
            }), 400
        
        # Get role-specific configuration
        config = ROLE_CONFIG[role]
        collection = collections[role]
        images_folder = config["images_folder"]
        default_role = config["default_role"]
        
        # Retrieve chunks using role-specific collection
        chunks = retrieve_chunks(query, default_role, collection, top_k)
        step_no = extract_step_number(query)
        
        # Handle no results
        if step_no is not None and not chunks:
            return jsonify({
                "success": False, 
                "error": f"Step {step_no} does not exist for role '{role}'."
            }), 404
        
        if not chunks:
            return jsonify({
                "success": False, 
                "error": f"No relevant information found for role '{role}'."
            }), 404
        
        # Build prompt and generate answer
        prompt = build_prompt(query, chunks, role)
        llm_answer = generate_gemini(prompt)
        
        # Embed images from role-specific folder
        llm_answer_html = embed_images_in_html(llm_answer, images_folder,role)
        
        return jsonify({
            "success": True,
            "role": role,
            "query": query,
            "answer": {"text": llm_answer_html},
        }), 200
    
    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


# -----------------------------
# 7. Run App
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000,debug=True)