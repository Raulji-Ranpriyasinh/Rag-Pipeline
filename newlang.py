from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import os
import re
import traceback
from typing import Optional, List
from typing_extensions import TypedDict
from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph, END

# -----------------------------
# 1. App Initialization
# -----------------------------
app = Flask(__name__)
CORS(app)

# -----------------------------
# 2. Config
# -----------------------------
BASE_DIR = os.getcwd()
DB_PATH = "RagDB"

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

for role_data in ROLE_CONFIG.values():
    os.makedirs(role_data["images_folder"], exist_ok=True)

# -----------------------------
# 3. DB + Embeddings
# -----------------------------
chroma_client = chromadb.PersistentClient(
    path=DB_PATH,
    settings=Settings(allow_reset=False)
)

collections = {}
for role, config in ROLE_CONFIG.items():
    collections[role] = chroma_client.get_collection(config["collection_name"])

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------
# 4. LangChain LLM
# -----------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2
)

prompt_template = PromptTemplate(
    input_variables=["context", "query", "role"],
    template="""
You are a {role} user-guide assistant.

Use ONLY the retrieved context. Ignore unrelated sections.
Always cite like [1], [2].

If image appears like [IMG:filename], include it.

CONTEXT:
{context}

User Question:
{query}

Answer:
"""
)

# -----------------------------
# 5. TypedDict State  ← KEY FIX
# -----------------------------
class State(TypedDict, total=False):
    query: str
    role: str
    chunks: List[dict]
    prompt: str
    answer: str
    final_answer: str

# -----------------------------
# 6. Utility Functions
# -----------------------------
def extract_step_number(query: str) -> Optional[int]:
    match = re.search(r"\bstep\s*(\d+)\b", query.lower())
    return int(match.group(1)) if match else None


def retrieve_chunks(query: str, role: str, collection, top_k: int = 3) -> List[dict]:
    step_no = extract_step_number(query)

    if step_no is not None:
        res = collection.get(
            where={"$and": [{"role": role}, {"order": step_no}]},
            include=["documents", "metadatas"]
        )

        chunks = []
        for doc, meta in zip(res["documents"], res["metadatas"]):
            chunks.append({
                "text": doc,
                "meta": meta,
                "order": meta.get("order", 0),
                "part": meta.get("part", 1),
            })

        chunks.sort(key=lambda x: x["part"])
        return chunks

    # Semantic search
    q_emb = embedding_model.encode([query])[0].tolist()

    res = collection.query(
        query_embeddings=[q_emb],
        n_results=top_k,
        where={"role": role},
        include=["documents", "metadatas"]
    )

    chunks = []
    for doc, meta in zip(res["documents"][0], res["metadatas"][0]):
        chunks.append({
            "text": doc,
            "meta": meta,
            "order": meta.get("order", 0),
            "part": meta.get("part", 1),
        })

    return chunks


# def embed_images_in_html(text: str, images_folder: str, role: str) -> str:
#     base_url = request.host_url.rstrip("/")
#     available_images = {f.lower(): f for f in os.listdir(images_folder)}

#     pattern = re.compile(r'\[(?:IMG|Image):\s*([^\]]+?)\]', re.IGNORECASE)

#     def replacer(match):
#         name = match.group(1).strip().lower()
#         if name in available_images:
#             file = available_images[name]
#             return f'<img src="{base_url}/images/{role}/{file}" style="max-width:800px;">'
#         return f"(Missing image: {name})"

#     return pattern.sub(replacer, text)
def embed_images_in_html(text: str, images_folder: str, role: str) -> str:
    base_url = request.host_url.rstrip("/")
    
    # Normalize keys: lowercase + remove ALL spaces (handles "tec _7.jpg" → "tec_7.jpg")
    available_images = {f.lower().replace(" ", ""): f for f in os.listdir(images_folder)}

    pattern = re.compile(r'\[(?:IMG|Image):\s*([^\]]+?)\]', re.IGNORECASE)

    def replacer(match):
        img_name = match.group(1).strip()
        img_key = img_name.lower().replace(" ", "")  # normalize query the same way
        if img_key in available_images:
            f_real = available_images[img_key]
            return f'<img src="{base_url}/images/{role}/{f_real}" alt="{f_real}" style="max-width:800px;">'
        return f"(Missing image: {img_name})"

    return pattern.sub(replacer, text)
# -----------------------------
# 7. LangGraph Nodes
# -----------------------------
def retrieve_node(state: State) -> State:
    query = state["query"]
    role = state["role"]
    chunks = retrieve_chunks(query, role, collections[role])
    return {"chunks": chunks}          # Return only new/updated keys


def prompt_node(state: State) -> State:
    chunks = state["chunks"]
    query = state["query"]
    role = state["role"]

    context = "\n\n".join(
        [f"[{i+1}] {c['text']}" for i, c in enumerate(chunks)]
    )

    prompt = prompt_template.format(
        context=context,
        query=query,
        role=role
    )

    return {"prompt": prompt}          # Return only new/updated keys


def llm_node(state: State) -> State:
    response = llm.invoke(state["prompt"]).content
    return {"answer": response}        # Return only new/updated keys


def image_node(state: State) -> State:
    role = state["role"]
    answer = embed_images_in_html(
        state["answer"],
        ROLE_CONFIG[role]["images_folder"],
        role
    )
    return {"final_answer": answer}    # Return only new/updated keys

# -----------------------------
# 8. Build Graph
# -----------------------------
builder = StateGraph(State)

builder.add_node("retrieve", retrieve_node)
builder.add_node("prompt", prompt_node)
builder.add_node("llm", llm_node)
builder.add_node("image", image_node)

builder.set_entry_point("retrieve")

builder.add_edge("retrieve", "prompt")
builder.add_edge("prompt", "llm")
builder.add_edge("llm", "image")
builder.add_edge("image", END)        # Explicit END edge

graph = builder.compile()

# -----------------------------
# 9. Routes
# -----------------------------
@app.route("/images/<role>/<path:filename>")
def serve_image(role, filename):
    if role not in ROLE_CONFIG:
        return "Invalid role", 400
    folder = ROLE_CONFIG[role]["images_folder"]
    return send_from_directory(folder, filename)


@app.route("/query", methods=["POST"])
def query():
    try:
        data = request.get_json()

        query_text = data.get("query")
        role = data.get("role")

        if not query_text or not role:
            return jsonify({"error": "query and role required"}), 400

        if role not in ROLE_CONFIG:
            return jsonify({"error": "invalid role"}), 400

        result = graph.invoke({
            "query": query_text,
            "role": role
        })

        return jsonify({
            "success": True,
            "answer": result["final_answer"]
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# -----------------------------
# 10. Run
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True, port=5000)