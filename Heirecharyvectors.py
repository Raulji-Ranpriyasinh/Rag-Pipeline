import re
from docx import Document
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# =============================
# CONFIG
# =============================
DOC_PATH = r"D:\Playground\API\RAG MEDICAL\Dedicated User Guides\Teacher_Guide.docx"
DB_PATH = "RagDB"
# COLLECTION_NAME = "superadmin_user_guide"
# ROLE = "Super Admin"
# COLLECTION_NAME = "admin_user_guide"
# ROLE = "Admin"
COLLECTION_NAME = "teacher_user_guide"
ROLE = "Teachers"


MAX_CHARS = 1500
OVERLAP_CHARS = 200

# =============================
# UTILS
# =============================
def safe(value, default):
    return value if value not in (None, "") else default

def get_para_font_size(para):
    if para.runs and para.runs[0].font.size:
        return para.runs[0].font.size.pt
    return 0.0

def extract_step_number(h2_text):
    if not h2_text:
        return -1
    match = re.search(r"step[\s\-]*(\d+)", h2_text.lower())
    return int(match.group(1)) if match else -1

def split_long_chunk(text):
    parts = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + MAX_CHARS, n)
        parts.append(text[start:end].strip())
        if end == n:
            break
        start = max(0, end - OVERLAP_CHARS)

    return parts

# =============================
# LOAD DOC
# =============================
doc = Document(DOC_PATH)

chunks = []
current_h1 = None
current_h2 = None
h1_content, h2_content = [], []

# =============================
# PARSE
# =============================
for para in doc.paragraphs:
    text = para.text.strip()
    if not text:
        continue

    style = para.style.name
    font_size = get_para_font_size(para)

    # ---- H1 ----
    if style == "Heading 1" or font_size == 20:
        if current_h2 and h2_content:
            chunks.append({
                "text": f"{current_h1}\n{current_h2}\n" + "\n".join(h2_content),
                "h1": current_h1,
                "h2": current_h2,
                "level": "H2",
                "order": extract_step_number(current_h2)
            })
            h2_content = []

        if current_h1 and h1_content:
            chunks.append({
                "text": f"{current_h1}\n" + "\n".join(h1_content),
                "h1": current_h1,
                "h2": None,
                "level": "H1",
                "order": -1
            })
            h1_content = []

        current_h1 = text
        current_h2 = None

    # ---- H2 ----
    elif style == "Heading 2" or font_size == 16:
        if current_h2 and h2_content:
            chunks.append({
                "text": f"{current_h1}\n{current_h2}\n" + "\n".join(h2_content),
                "h1": current_h1,
                "h2": current_h2,
                "level": "H2",
                "order": extract_step_number(current_h2)
            })
        current_h2 = text
        h2_content = []

    # ---- BODY ----
    else:
        if current_h2:
            h2_content.append(text)
        elif current_h1:
            h1_content.append(text)

# =============================
# FLUSH
# =============================
if current_h2 and h2_content:
    chunks.append({
        "text": f"{current_h1}\n{current_h2}\n" + "\n".join(h2_content),
        "h1": current_h1,
        "h2": current_h2,
        "level": "H2",
        "order": extract_step_number(current_h2)
    })

if current_h1 and h1_content:
    chunks.append({
        "text": f"{current_h1}\n" + "\n".join(h1_content),
        "h1": current_h1,
        "h2": None,
        "level": "H1",
        "order": -1
    })

print(f"✅ Semantic chunks before split: {len(chunks)}")

# =============================
# SPLIT
# =============================
final_chunks = []

for c in chunks:
    if c["level"] == "H2" and len(c["text"]) > MAX_CHARS:
        for i, part in enumerate(split_long_chunk(c["text"]), start=1):
            final_chunks.append({**c, "text": part, "part": i})
    else:
        final_chunks.append({**c, "part": 1})

print(f"✅ Total chunks after split: {len(final_chunks)}")

# =============================
# EMBEDDINGS
# =============================
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode([c["text"] for c in final_chunks], show_progress_bar=True)

# =============================
# CHROMA INGEST
# =============================
client = chromadb.PersistentClient(
    path=DB_PATH,
    settings=Settings(allow_reset=True)
)

if COLLECTION_NAME in [c.name for c in client.list_collections()]:
    client.delete_collection(COLLECTION_NAME)

collection = client.create_collection(name=COLLECTION_NAME)

for idx, (c, e) in enumerate(zip(final_chunks, embeddings)):
    collection.add(
        ids=[str(idx)],
        documents=[c["text"]],
        embeddings=[e.tolist()],
        metadatas=[{
            "role": ROLE,
            "topic": safe(c["h1"], "General"),
            "subtopic": safe(c["h2"], "Overview"),
            "level": c["level"],
            "order": int(c["order"]),   # ALWAYS INT
            "part": int(c["part"])
        }]
    )

print("🚀 Vectorization completed successfully")
