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
# 2. Configuration
# -----------------------------
BASE_DIR = os.getcwd()
IMAGES_FOLDER = os.path.join(BASE_DIR, "images", "admin_img")
DB_PATH = "RagDB"
COLLECTION_NAME = "admin_user_guide"

ALLOWED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}

os.makedirs(IMAGES_FOLDER, exist_ok=True)

# -----------------------------
# 3. ChromaDB + Embedding Model
# -----------------------------
chroma_client = chromadb.PersistentClient(
    path=DB_PATH,
    settings=Settings(allow_reset=False)
)

collection = chroma_client.get_collection(COLLECTION_NAME)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------
# 4. Utility Functions
# -----------------------------
def extract_step_number(query: str):
    match = re.search(r"\bstep\s*(\d+)\b", query.lower())
    return int(match.group(1)) if match else None


def retrieve_chunks(query: str, role: str, top_k: int = 3):
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
        if "student" in q_lower and "student" in (h1 + h2).lower(): bonus += 0.2
        if "teacher" in q_lower and "teacher" in (h1 + h2).lower(): bonus += 0.2
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


def build_prompt(query: str, chunks: list):
    context_blocks = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk["meta"]
        header = f"[{i}] {chunk['h1']}"
        if chunk["h2"]: header += f" → {chunk['h2']}"
        header += f" | StepOrder={meta.get('order')} Part={meta.get('part')}"
        context_blocks.append(f"{header}\n{chunk['text']}")
    context_str = "\n\n".join(context_blocks)

    return f"""
    
You are an admin user-guide assistant.
Use ONLY the retrieved context. Ignore unrelated sections. Always cite the section(s) used (e.g., [1], [2]).
If an image appears as [IMG:filename.ext], include it at the correct step.
Only create numbered steps if the context explicitly lists them as Step 1, Step 2, etc.
Do not convert descriptive summaries into steps.
Do not invent actions, parameters, workflows, or examples.
If no explicit steps exist, respond with a concise descriptive paragraph summarizing the relevant content from the context.
Ignore all information in the context that is unrelated to the user’s query.
Keep the answer direct and focused on the question.

CONTEXT:
{context_str}

User question: {query}

Write a short, step-by-step answer.

"""


def embed_images_in_html(text):
    # Use request.host_url correctly
    base_url = request.host_url.rstrip("/")
    pattern = re.compile(r'\[(?:Image|IMG):\s*([^\]]+?)\]', re.IGNORECASE)

    # Preload available images case-insensitive
    available_images = {f.lower(): f for f in os.listdir(IMAGES_FOLDER)}

    def replacer(match):
        img_name = match.group(1).strip()
        img_key = img_name.lower()
        if img_key in available_images:
            f_real = available_images[img_key]
            return f'<img src="{base_url}/images/{f_real}" alt="{f_real}" style="max-width:800px;">'
        else:
            return f'❌ Image not found: {img_name}'

    return pattern.sub(replacer, text)



@app.route("/images/<path:filename>", methods=["GET"])
def serve_image(filename):
    requested_lower = filename.lower()
    try:
        for f in os.listdir(IMAGES_FOLDER):
            if f.lower() == requested_lower:
                return send_from_directory(IMAGES_FOLDER, f)
        return jsonify({"success": False, "error": f"Image not found: {filename}"}), 404
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

# -----------------------------
# RAG Query Endpoint
# -----------------------------
@app.route("/adminquery", methods=["POST"])
def query_guide():
    try:
        data = request.get_json()
        if not data or "query" not in data:
            return jsonify({"success": False, "error": "Missing 'query' field"}), 400

        query = data["query"]
        role = data.get("role", "admin")
        top_k = int(data.get("top_k", 3))
        top_k = max(1, min(top_k, 10))

        chunks = retrieve_chunks(query, role, top_k)
        step_no = extract_step_number(query)

        if step_no is not None and not chunks:
            return jsonify({"success": False, "error": f"Step {step_no} does not exist."}), 404
        if not chunks:
            return jsonify({"success": False, "error": "No relevant information found."}), 404

        prompt = build_prompt(query, chunks)
        llm_answer = generate_gemini(prompt)
        llm_answer_html = embed_images_in_html(llm_answer)

        return jsonify({
            "success": True,
            "query": query,
            "answer": {"text": llm_answer_html},
        }), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500



# -----------------------------
# 10. Run App
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5003)
