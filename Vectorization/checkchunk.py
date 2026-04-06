# =============================
# VERIFY CHUNKS IN CHROMA (with parts)
# =============================
import chromadb
from chromadb.config import Settings

DB_PATH = "RagDB"
COLLECTION_NAME = "superadmin_user_guide"

client = chromadb.PersistentClient(
    path=DB_PATH,
    settings=Settings()
)

collection = client.get_collection(COLLECTION_NAME)

# ids cannot be in include for .get()
results = collection.get(
    include=["documents", "metadatas"]   # <-- fixed
)

docs = results["documents"]
metas = results["metadatas"]

print(f"📊 Total chunks: {len(docs)}")
print("=" * 100)

records = []
for i, (doc, meta) in enumerate(zip(docs, metas)):
    records.append({
        "idx": i,                        # local index as id
        "doc": doc,
        "topic": meta.get("topic", "General"),
        "subtopic": meta.get("subtopic", "Overview"),
        "level": meta.get("level", "N/A"),
        "order": meta.get("order", -1),
        "part": meta.get("part", 1),
        "role": meta.get("role", "N/A"),
    })

# Sort by topic, subtopic, order, part
records.sort(key=lambda r: (r["topic"], r["subtopic"], r["order"], r["part"]))

current_topic = None
current_subtopic = None

for r in records:
    if r["topic"] != current_topic or r["subtopic"] != current_subtopic:
        current_topic = r["topic"]
        current_subtopic = r["subtopic"]
        print("\n" + "=" * 100)
        print(f"📂 Parent Topic (H1): {current_topic}")
        print(f"📋 Subtopic (H2): {current_subtopic}")
        print("=" * 100)

    print(f"\n{'-'*100}")
    print(
        f"🔹 CHUNK Idx: {r['idx']} | "
        f"Order: {r['order']} | "
        f"Level: {r['level']} | "
        f"Part: {r['part']}"
    )
    print(f"👤 Role: {r['role']}")
    print("-" * 100)
    print("📄 CONTENT:")
    print(r["doc"])
    print("-" * 100)

print("\n" + "=" * 100)
print("✅ Verification complete (grouped by topic/subtopic, showing parts in order)!")
