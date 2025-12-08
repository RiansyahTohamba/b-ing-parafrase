from sentence_transformers import SentenceTransformer

# Muat model (yang ringan dan cepat)
model = SentenceTransformer("all-MiniLM-L6-v2")

kalimat = [
    "Kucing sedang tidur di sofa.",
    "Seekor anjing berlari di taman."
]

# Hasil embedding berupa array float berukuran 384
embeddings = model.encode(kalimat)

for i, emb in enumerate(embeddings):
    print(f"Kalimat {i+1} → dimensi: {len(emb)}")
    print(emb[:10], "...")   # tampilkan 10 angka pertama
