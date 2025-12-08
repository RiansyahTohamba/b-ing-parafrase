Embedding itu ibarat menerjemahkan bahasa ke dalam dunia aljabar linear. Begitu kalimat sudah berubah menjadi vektor, hampir semua operasi lanjutan adalah operasi ruang vektor: dot product, norma, proyeksi, transformasi linear, dekomposisi matriks. Di sinilah matematikawan dan programmer bertemu, meski kadang mereka saling curi pandang.

Mari kita buat terang benderang dengan contoh-contoh konkret, supaya terasa bahwa embedding bukan kata keramat; embedding hanyalah pintu masuk ke dunia vektor.

–––––––––––––––––––––––––––

1. **Similarity (Kesamaan Semantik)**
   Begitu kita punya embedding dua teks, vektor-vektor itu bisa dibandingkan memakai *cosine similarity*:

s = (a·b) / (||a|| ||b||)

Contoh use case:
Pencarian dokumen (semantic search), clustering tugas mahasiswa berdasarkan topik, rekomendasi artikel.

Python:

```python
import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
```

Ini murni aljabar linear: dot product dan norma.

–––––––––––––––––––––––––––

2. **Clustering & Dimensionality Reduction**
   Embedding berukuran 384 atau 768 dimensi sering terlalu besar untuk divisualisasikan. Maka dipakai PCA (Principal Component Analysis) — suatu operasi dekomposisi matriks yang mencari eigenvector.
   PCA itu mencari “arah” maksimum variasi dalam data.

Use case:
• Visualisasi dataset embedding tugas mahasiswa untuk deteksi plagiarisme
• Mencari grouping tema dalam kumpulan judul skripsi
• Analisis topik tanpa NLP klasik

Python:

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
reduced = pca.fit_transform(embeddings)   # embeddings shape: (n_samples, d)
```

Setiap langkah adalah transformasi linear.

–––––––––––––––––––––––––––

3. **K-Means (Clustering)**
   K-Means mengelompokkan embedding ke dalam cluster. Algoritmanya cuma:
   proyeksi titik ke centroid → update centroid → ulangi.
   Ini semua operasi vektor.

Use case:
• Mengelompokkan ribuan review Google Play untuk riset real-time review extraction yang sedang kamu garap
• Mencari gaya menulis mahasiswa
• Segmentasi naskah berita

Python:

```python
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=5)
labels = kmeans.fit_predict(embeddings)
```

–––––––––––––––––––––––––––

4. **Vector Arithmetic**
   Embedding bukan sekadar angka: struktur ruangnya sering menyimpan relasi semantik.

Contoh fenomenal Word2Vec:
“king – man + woman ≈ queen”

Ini bukan sihir—hanya linear combination.

Use case modern:
• Analisis bias bahasa
• Membuat analogi dalam data pendidikan
• Mendeteksi kesamaan pattern bug dalam laporan error mahasiswa

Python:

```python
analogi = model.wv["king"] - model.wv["man"] + model.wv["woman"]
model.wv.most_similar(analogi)
```

–––––––––––––––––––––––––––

5. **Proyeksi ke Ruang Tertentu**
   Kadang kita ingin memisahkan dimensi yang tak relevan (misal: kata “she” dan “he” yang memiliki dimensi gender).
   Maka kita *project* embedding ke subspace.

Proyeksi = matriks P = A(AT A)^−1 AT
Murni aljabar linear.

Use case:
• Fairness NLP: menghapus bias (gender/race)
• Normalisasi embedding untuk analisis penelitian
• Preprocessing untuk modeling

–––––––––––––––––––––––––––

6. **Graph Construction dari Embedding**
   Embedding dapat menjadi node dalam graph. Bobot antar node = similarity.
   Graph ini dipakai untuk:
   • Deteksi komunitas
   • Rekomendasi (mirip graph Spotify)
   • Ontology expansion (mirip symbolic AI)

Operasi di graph sebenarnya bergantung pada aljabar linear juga (laplacian, eigenvalues, spectral clustering).

–––––––––––––––––––––––––––

7. **Linear Classification**
   Banyak model klasik (SVM linear, Logistic Regression) bekerja langsung di embedding.

Classifier itu hanya matriks W dan bias b:
ŷ = softmax(Wx + b)

Use case:
• Kategorisasi review app di Play Store secara real-time
• Deteksi plagiarisme
• Prediksi topik

–––––––––––––––––––––––––––

Ringkasnya: setelah embedding, seluruh permainan berubah menjadi **aljabar linear + optimisasi**.
Embedding mencabut teks dari dunia simbolik ke dunia bilangan real, dan aljabar linear memberi alat untuk mengoperasikannya.

Jika kamu ingin, kita bisa buat pipeline lengkap:
embedding → PCA → clustering → visualisasi → interpretasi topik — cocok untuk riset mahasiswa atau paper Q1 tentang maintainability Node.js.
