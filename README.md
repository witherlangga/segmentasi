# Leaf Segmentation & Health Classification Web App
## Struktur & entry points utama
- **Aplikasi web / UI**
  - app.py — **entry point** Flask. Menangani upload (multiple), menyimpan file (uploads), memanggil `segment_leaf()` dan merender result.html.
  - index.html, result.html — UI untuk upload dan tampilan hasil.

- **Segmentasi & fitur**
  - kmeans.py
    - Fungsi utama: `segment_leaf(image_path, K=10)`
    - Proses: hapus background (rembg) → buat mask → refine mask (OpenCV) → ambil piksel daun → KMeans clustering → rule-based color classification → buat gambar segmentasi dan per-cluster → buat bar chart → (opsional) panggil RF untuk prediksi klaster dan per-daun → kembalikan dict hasil.

- **Klasifikasi (Random Forest)**
  - rf_classifier.py
    - `extract_cluster_features(image_path, K=10)` → mengekstrak fitur per-klaster: [R,G,B, percentage]
    - `build_training_set(dataset_root, K)` → scan dataset (subfolder: "sakit", "sehat", "kurang sehat") dan buat X,y
    - `train_model(force=False)` → latih RandomForest, simpan model rf_leaf.pkl dan mapping label_mapping.json
    - `load_model()` → load model & mapping (mengembalikan `(clf, inv)`), `predict_clusters()` dan `predict_leaf()` untuk prediksi
    - `get_confidence_threshold()` membaca env var `RF_CONF_THRESHOLD` (default 70%)

- **Bantuan / util scripts**
  - run_training.py (root) dan run_training.py — wrapper memanggil `train_model(force=True)`
  - check_preprocess.py — cek ekstraksi fitur pada dataset (menulis `preprocess_check.json`)
  - inspect_training.py — inspeksi X,y dan coba latih
  - dump_training_info.py — dump `training_info.json`
  - generate_report.py — membuat laporan .docx menggunakan hasil uji (test_results_summary.json) dan gambar di Test

---

## Urutan eksekusi end-to-end (nomor langkah)
1. Pengguna membuka index.html dan meng-upload satu atau lebih gambar, menentukan `k` (jumlah klaster).
2. app.py menerima POST:
   - Simpan file di uploads
   - Pastikan model RF tersedia: `load_model()` → jika None, `train_model()` dipanggil (otomatis, kecuali gagal)
3. Untuk tiap file: `segment_leaf(filepath, K)`:
   - Baca bytes gambar → `rembg.remove()` untuk menghilangkan background → hasil RGBA
   - Buat mask alpha → refine (morphology + largest connected component)
   - Ambil piksel daun → KMeans clustering (K)
   - Hitung centroid (R,G,B) tiap klaster → rule-based `classify_leaf_color()` untuk label sementara
   - Hitung persentase tiap klaster, gambar tersegmentasi (`static/results/seg_{name}.png`), tiap klaster (`cluster_{i}_{name}.png`) dan bar chart (`bar_{name}.png`)
   - Jika model RF ada, `predict_clusters()` (per-klaster) dan `predict_leaf()` (agregasi probabilitas bobot) → buat `conclusion` dan `short_conclusion`
4. app.py mengumpulkan hasil semua gambar:
   - Jika >1 gambar, buat ringkasan perbandingan (jumlah & rata‑rata probabilitas)
5. Render result.html menampilkan:
   - Gambar asli, hasil segmentasi, grid klaster, tabel data numerik, grafik distribusi, kesimpulan singkat per gambar, dan summary antar-gambar

---

## Output & lokasi file
- Uploads: `static/uploads/<filename>`
- Hasil segmentasi: `static/results/seg_<filename>.png`
- Per-cluster images: `static/results/cluster_<i>_<filename>.png`
- Bar chart: `static/results/bar_<filename>.png`
- Model: rf_leaf.pkl
- Mapping label: label_mapping.json
- Laporan & metadata: `preprocess_check.json`, `training_info.json`, test_results_summary.json dan Implementasi

---

## Cara menjalankan
- Install dependencies: `pip install -r requirements.txt`
- Jalankan web app: `python app.py` (buka http://127.0.0.1:5000)
- Latih model manual: `python run_training.py` (atau `python scripts/run_training.py`)
- Cek preprocess: `python scripts/check_preprocess.py`
- Buat laporan: `python scripts/generate_report.py`

---

## Catatan penting & keterbatasan
- Dependensi utama: `rembg` (menggunakan model background removal), OpenCV, scikit‑learn.
- Fitur yang dipakai untuk RF hanyalah kombinasi warna (R,G,B) + persentase per-klaster — tidak ada fitur bentuk/tekstur yang kompleks. Ini membatasi kemampuan klasifikasi.
- Rule-based color classification cukup sederhana (hanya memeriksa perbandingan R/G/B) → bisa salah di kondisi pencahayaan buruk.
- `train_model()` otomatis akan menolak jika dataset kosong; fungsi mencoba melatih otomatis jika model hilang, tetapi ini bisa gagal jika dataset tidak lengkap.
- Ambang keyakinan dapat dikonfigurasi dengan env var `RF_CONF_THRESHOLD`.

---

## Rekomendasi singkat
- Tambahkan fitur bentuk/tekstur (Laplacian, Haralick, area/perimeter) untuk meningkatkan RF.
- Pertimbangkan model segmentasi yang lebih kuat (mis. U-Net) untuk kondisi background/lighting yang sulit.
- Perbaiki pipeline training: pre-processing standar (resize, color normalization) dan validasi lebih ketat.
