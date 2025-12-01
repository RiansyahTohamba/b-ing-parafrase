Bayangkan transformasi linear di Torch sebagai *mantra teleportasi makna* antar NPC dalam game. NPC yang tadinya cuma kumpulan teks hambar berubah menjadi entitas yang bisa saling “mengerti” berkat permainan ruang vektor. Linear algebra + NLP = mesin dialog yang hidup.

Mari kita kaitkan pilar-pilar transformasi linear tadi secara langsung dengan pipeline NLP NPC di game—tanpa mistik, tetap rasional, tetapi sambil menikmati keanehan konsep “ruang makna”.

---

## 1. Representasi Dasar: NPC Butuh “Tubuh Vektor”

Sebelum NPC bicara, kalimat mereka harus diubah jadi angka. Ini dilakukan dengan *embedding*, yang dasarnya adalah transformasi linear terlatih.

Dalam Torch:

```python
embedding = torch.nn.Embedding(num_embeddings=5000, embedding_dim=128)
token_ids = torch.tensor([5, 18, 233, 90])
vecs = embedding(token_ids)
```

Embedding itu sebenarnya tabel besar yang, ketika kamu ambil satu barisnya, kamu sedang melakukan transformasi linear tak eksplisit: memilih satu vektor representasi dari ruang makna.

NPC yang punya kalimat berbeda akan punya vektor berbeda. Mereka sudah punya “bentuk” dalam ruang semantik.

---

## 2. Linear Layer = Otak Mini NPC

Setiap NPC bisa diberi *brain module* berupa layer linear yang belajar:

* cara memilih respons,
* cara memprediksi emosi,
* cara mengubah tone bicara.

Misalnya, menentukan *intent* dari dialog pemain:

```python
brain = torch.nn.Linear(128, 10)  # 10 intent classes
intent_logits = brain(vecs.mean(dim=0))
```

Layer linear ini bekerja persis seperti transformasi geometri:
NPC memproyeksikan makna kalimat pemain dari ruang 128D ke ruang 10D.

Seperti teleportasi antardimensi tetapi matematis.

---

## 3. Interaksi Antar NPC = Transformasi Linear Antar Ruang Makna

Di game RPG, NPC sering saling merespons, meski scripted. Transformasi linear membuat respons ini *tidak sekadar hard-coded*.

Misalnya kamu punya dua NPC:

* NPC A: penjaga kota
* NPC B: pedagang licik

Masing-masing punya embedding persona:

```python
npcA = torch.randn(128)
npcB = torch.randn(128)
```

Lalu kamu punya modul dialog yang menggabungkan konteks:

```python
W = torch.nn.Linear(128 * 2, 128)  # transformasi interaksi
interaction = W(torch.cat([npcA, npcB]))
```

Transformasi linear ini membentuk “ruang interaksi”, di mana:

* makna penjaga,
* makna pedagang,
* dan situasi percakapan

semuanya direduksi jadi vektor baru yang dipakai untuk memilih dialog *real-time*.

Perpaduan ruang seperti ini adalah alasan mengapa model modern bisa menghasilkan percakapan yang terasa organik.

---

## 4. LSTM / GRU / Transformer = Kerajaan Transformasi Linear

Semua struktur NLP ilmiah berdiri di atas transformasi linear berulang.

Di Torch, misalnya Transformer:

```python
layer = torch.nn.TransformerEncoderLayer(
    d_model=128, nhead=4
)
output = layer(vecs.unsqueeze(1))
```

Di dalam satu layer Transformer ada:

* empat matriks linear untuk Q, K, V, O,
* dua linear layer di feed-forward block,
* layernorm (juga berbasis transformasi skala).

NPC bisa diberi otak Transformer kecil untuk:

* menafsir dialog pemain,
* mempertahankan konteks,
* menyusun respons unik.

Jadi *sistem percakapan NPC modern = orkestra transformasi linear yang saling menopang*.

---

## 5. Contoh Mini: NPC Menjawab Berdasarkan Transformasi Linear

Ini gambaran minimalis tetapi mencerminkan esensi.

Misal kita buat *semantic matcher* sederhana—NPC memilih respons berdasarkan kemiripan vektor:

```python
# Kalimat pemain (sudah diekstrak embeddingnya)
player = torch.randn(128)

# Tiga kemungkinan dialog NPC (embedding)
npc_dialogs = torch.randn(3, 128)

# Transformasi linear untuk memproyeksikan ke ruang kesesuaian
T = torch.nn.Linear(128, 128)
player_T = T(player)
dialogs_T = T(npc_dialogs)

# NPC memilih respons paling dekat
scores = torch.cosine_similarity(player_T, dialogs_T)
choice = torch.argmax(scores)
```

Transformasi T membuat NPC punya gaya bicara unik:

* Penjaga bisa lebih “formal”
* Pedagang lebih “ngemis-ngemis”
* Penyihir lebih “misterius”

Semua hanya dengan mengubah matriks transformasi.

Inilah seni geometri makna.

---

## 6. Menuju Level Ahli: Mengajar NPC untuk Mengingat dan Berkembang

Jika kamu mengkombinasikan:

* embedding,
* beberapa linear layer,
* modul konteks (LSTM/Transformer),
* head output yang juga linear,

maka sistem dialog NPC hampir setara dengan model bahasa mini.

Torch memfasilitasi semuanya.

---

Transformasi linear bukan lagi alat matematika abstrak; ia menjadi infrastruktur komunikasi antara entitas digital yang hidup dalam dunia game. Dari satu vektor ke vektor lain, mereka membangun budaya kecil mereka sendiri.
