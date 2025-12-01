Bagus — mari kita bangun contoh lengkap: sebuah mini-Flask app yang **menggunakan SQLite (SQLAlchemy)** untuk menyimpan baris dialog NPC, **menghitung representasi vektor** (menggunakan `TfidfVectorizer` dari scikit-learn sebagai embedding ringan/stand-in), menyimpan embedding ke database (sebagai BLOB), dan memilih respons NPC berdasarkan **cosine similarity** pada ruang vektor.
Ini mudah dijalankan secara lokal, jelas memperlihatkan peran *vector space* pada pemilihan respons, dan masih kompatibel bila nanti ingin mengganti `Tfidf` dengan model embedding deep (mis. `sentence-transformers`).

Kode utuh di satu file `app.py` — salin, pasang dependencies, dan jalankan.

# Kode: `app.py`

```python
# app.py
from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import LargeBinary
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import io
import os

# --- Config ---
DATABASE_URL = "sqlite:///npc_dialogs.db"
TOP_K = 1  # berapa respons teratas dikembalikan

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URL
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# --- Models ---
class NPCLine(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    npc_name = db.Column(db.String, nullable=False)
    text = db.Column(db.Text, nullable=False)
    # optional metadata
    season = db.Column(db.String, nullable=True)
    tags = db.Column(db.String, nullable=True)
    # store embedding as binary blob (numpy array)
    embedding = db.Column(LargeBinary, nullable=True)

    def to_dict(self):
        return {
            "id": self.id,
            "npc_name": self.npc_name,
            "text": self.text,
            "season": self.season,
            "tags": self.tags
        }

# --- Vector space components (in-memory) ---
# We'll fit a TF-IDF on all NPC lines' texts. For small dataset this is fine.
vectorizer = TfidfVectorizer()

def np_to_blob(a: np.ndarray) -> bytes:
    bio = io.BytesIO()
    # save array in npy format
    np.save(bio, a)
    bio.seek(0)
    return bio.read()

def blob_to_np(b: bytes) -> np.ndarray:
    bio = io.BytesIO(b)
    bio.seek(0)
    return np.load(bio)

def build_vector_index():
    """
    Read all NPC lines, fit vectorizer, store embeddings back to DB.
    Call this after initial data insertions or when dataset changes.
    """
    all_lines = NPCLine.query.all()
    texts = [l.text for l in all_lines]
    if len(texts) == 0:
        return None, None

    # fit TF-IDF (embedding)
    X = vectorizer.fit_transform(texts)  # sparse matrix
    # store each vector into DB
    for i, line in enumerate(all_lines):
        vec = X[i].toarray().astype(np.float32).reshape(-1)
        line.embedding = np_to_blob(vec)
    db.session.commit()
    return vectorizer, X

def load_embeddings_matrix():
    """
    Load embeddings from DB into an in-memory matrix (n_samples, dim).
    Returns list_of_lines, matrix (numpy)
    """
    all_lines = NPCLine.query.all()
    if not all_lines:
        return [], None
    embs = []
    for l in all_lines:
        if l.embedding is None:
            # if any missing, rebuild index
            build_vector_index()
            return load_embeddings_matrix()
        arr = blob_to_np(l.embedding)
        embs.append(arr)
    matrix = np.vstack(embs)  # shape (n_lines, dim)
    return all_lines, matrix

# --- Routes / API ---

@app.route('/init', methods=['POST'])
def init_db_route():
    """Initialize DB and optionally seed with demo lines."""
    db.create_all()
    data = request.json or {}
    seed = data.get('seed', False)
    if seed:
        # sample NPC lines
        sample = [
            {"npc_name": "Farmer Joe", "text": "Selamat pagi! Sudah memberi makan sapi hari ini?", "season": "spring"},
            {"npc_name": "Baker May", "text": "Ada roti hangat hari ini. Mau coba?", "season": "spring"},
            {"npc_name": "Mayor", "text": "Festival panen akan dimulai minggu depan.", "season": "fall"},
            {"npc_name": "Old Man", "text": "Dulu aku pernah berlayar ke pulau seberang.", "season": "summer"},
        ]
        for s in sample:
            l = NPCLine(npc_name=s['npc_name'], text=s['text'], season=s.get('season'))
            db.session.add(l)
        db.session.commit()
        build_vector_index()
        return jsonify({"status": "initialized and seeded", "seeded": len(sample)})
    else:
        return jsonify({"status": "initialized"})

@app.route('/npc', methods=['POST'])
def add_npc_line():
    """
    Add a new NPC dialog line.
    Body JSON: { "npc_name": "...", "text": "...", "season": "...", "tags": "..." }
    After insertion we rebuild embeddings (simple approach).
    """
    payload = request.json
    if not payload or 'npc_name' not in payload or 'text' not in payload:
        return jsonify({"error": "npc_name and text required"}), 400
    l = NPCLine(npc_name=payload['npc_name'], text=payload['text'],
                season=payload.get('season'), tags=payload.get('tags'))
    db.session.add(l)
    db.session.commit()
    # rebuild vector index (for small dataset ok)
    build_vector_index()
    return jsonify(l.to_dict()), 201

@app.route('/npc/<int:line_id>', methods=['GET'])
def get_npc_line(line_id):
    l = NPCLine.query.get_or_404(line_id)
    return jsonify(l.to_dict())

@app.route('/respond', methods=['POST'])
def respond():
    """
    Main endpoint: given player's utterance, return best NPC lines.
    Body JSON: { "player_text": "..." , optional filters: "npc_name", "season" }
    """
    payload = request.json or {}
    player_text = payload.get('player_text')
    if not player_text:
        return jsonify({"error": "player_text required"}), 400

    # load DB lines and embeddings
    lines, matrix = load_embeddings_matrix()
    if not lines or matrix is None:
        return jsonify({"error": "no npc lines in database"}), 400

    # vectorize player_text using existing vectorizer
    v = vectorizer.transform([player_text]).toarray().astype(np.float32)
    # compute cosine similarity
    sims = cosine_similarity(v, matrix)[0]  # shape (n_lines,)
    # pick top-K indices
    top_idx = np.argsort(-sims)[:TOP_K]
    results = []
    for idx in top_idx:
        l = lines[int(idx)]
        results.append({
            "npc": l.npc_name,
            "text": l.text,
            "score": float(sims[int(idx)])
        })
    return jsonify({"query": player_text, "results": results})

@app.route('/rebuild', methods=['POST'])
def rebuild_route():
    """Force re-build of vector index (recompute embeddings)."""
    vectorizer_local, X = build_vector_index()
    if X is None:
        return jsonify({"status": "no lines to index"})
    return jsonify({"status": "rebuilt", "n_lines": X.shape[0], "dim": X.shape[1]})

# --- Simple debug homepage ---
@app.route('/', methods=['GET'])
def home():
    return """
    <h3>NPC Vector Space Demo</h3>
    <p>Endpoints:</p>
    <ul>
      <li>POST /init  (json: { "seed": true })</li>
      <li>POST /npc  (json: { npc_name, text, season? })</li>
      <li>POST /respond (json: { player_text })</li>
      <li>POST /rebuild</li>
    </ul>
    """

if __name__ == '__main__':
    # create db file if not exists
    if not os.path.exists('npc_dialogs.db'):
        db.create_all()
    app.run(debug=True, host='0.0.0.0', port=5000)
```

# Cara pakai (singkat)

1. Buat virtualenv dan install deps:

   ```bash
   python -m venv venv
   source venv/bin/activate       # atau venv\Scripts\activate di Windows
   pip install flask sqlalchemy scikit-learn numpy
   ```
2. Simpan kode di `app.py` dan jalankan:

   ```bash
   python app.py
   ```
3. Inisialisasi dan seed contoh:

   ```bash
   curl -X POST http://127.0.0.1:5000/init -H "Content-Type: application/json" -d '{"seed": true}'
   ```
4. Tes respons:

   ```bash
   curl -X POST http://127.0.0.1:5000/respond -H "Content-Type: application/json" -d '{"player_text":"Apakah ada festival?"}'
   ```

   — server akan membalas baris NPC terdekat menurut TF-IDF / cosine similarity.

# Penjelasan singkat arsitektur & bagaimana vector space berperan

* **SQL (SQLite via SQLAlchemy)** menyimpan *master record* baris dialog (NPCLine). SQL bagus untuk query, metadata, manajemen konten, export/import.
* **Vector space (TF-IDF vectors)** memetakan setiap baris dialog ke titik di ruang berdimensi — memungkinkan menghitung *kemiripan semantik* (cosine similarity) antara input pemain dan teks NPC. Itu memberikan kemampuan generalisasi: pemain tidak harus memilih jawaban yang cocok secara literal, yang penting arti/konsepnya dekat.
* **Pilihan respons**: alih-alih match rule `IF player_said == "X" THEN respond "Y"`, kita mencari baris yang paling dekat dalam ruang vektor. Ini berguna untuk input bebas (ketik) atau sistem dialog semi-terbuka.
* **Penyimpanan embedding**: contoh menyimpan embedding di DB sebagai BLOB untuk membuatnya persist dan memungkinkan reload tanpa rekalkulasi penuh. (Untuk dataset besar, teknik lain lebih efisien: vector DB terpisah — FAISS, Annoy, Milvus — dan menyimpan DB metadata di SQL.)

# Perluasan dan catatan implementasi (praktis)

* `TfidfVectorizer` hanya contoh (mudah, tidak perlu internet). Untuk kualitas semantik lebih baik gunakan model embedding (mis. `sentence-transformers/all-MiniLM-L6-v2`) — lalu Anda perlu menyimpan vektor dimensi lebih besar dan mungkin gunakan approximate nearest neighbor (ANN) untuk pencarian cepat.
* Untuk NPC yang adaptif: simpan history interaksi pemain, gabungkan konteks (history) menjadi query vector (mis. average atau weighted) lalu cari respons.
* Untuk personality/style: tiap NPC punya *offset vector* atau filter pada hasil similarity untuk preferensi gaya.
* Performance: contoh ini melakukan rebuild `vectorizer` setelah setiap insert — ok untuk prototyping. Untuk produksi, gunakan incremental indexing / batch rebuild.

---

Kalau mau, berikut beberapa arah pengembangan yang bisa langsung saya tambahkan ke kode ini tanpa menunda:

* ganti TF-IDF ke `sentence-transformers` (butuh internet untuk download model)
* pakai FAISS atau Annoy untuk pencarian ANN (skalabilitas)
* tambahkan endpoint admin untuk edit / hapus dialog
* fitur konteks: gabungkan `player_history` ke query embedding

Mau saya tambahkan salah satu dari hal di atas sekarang? Saya bisa langsung modifikasi kode (mis. upgrade ke sentence-transformers) — pilih saja.
