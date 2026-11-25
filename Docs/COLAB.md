# Menjalankan Proyek di Google Colab

Sejak notebook dibuat satu file, menjalankannya di Colab jadi jauh lebih
sederhana: **tidak ada clone repo, tidak ada upload `src/`**. Semua kode ada di
dalam notebook.

---

## Yang perlu disiapkan

Cuma dua:

1. Notebook `notebooks/02_praproses.ipynb` — sudah ada di GitHub.
2. Akun Google. Tidak perlu Drive.

Dataset tidak perlu disiapkan manual — notebook mengunduhnya sendiri.

---

## Langkah menjalankan

1. Buka notebook dari GitHub di Colab:
   `https://colab.research.google.com/github/AhmadKevin000/Project-PCVK/blob/radita/notebooks/02_praproses.ipynb`
2. **Runtime → Run all**
3. Tunggu:

| Tahap | Waktu |
|---|---|
| Pasang `huggingface_hub` | ~10 detik |
| Unduh dataset 562 MB / 4049 berkas | ~2 menit |
| Bangun metadata + bagi fold | ~2 detik |
| Praproses 4049 citra (V0 + V2) | ~30 detik |
| Statistik 300 citra x 4 varian | ~40 detik |
| **Total** | **~3–4 menit** |

4. Kalau mau menyimpan hasilnya: **File → Save a copy in GitHub**. Ini yang
   membuat output ikut tersimpan dan bisa dilihat orang lain.

Kalau dijalankan di laptop, cell persiapan mencetak `Google Colab: False` lalu
berhenti — tidak ada yang diunduh.

---

## Apa yang hilang tiap sesi tutup

Colab menghapus seluruh isi `/content` setiap sesi berakhir. Jadi pahami dulu
mana yang perlu dijaga:

| Isi | Ukuran | Hilang saat sesi tutup? | Perlu disimpan ke Drive? |
|---|---|---|---|
| Dataset mentah (4049 BMP) | 562 MB | ya | tidak — unduh ulang ~2 menit |
| Cache `X_V0.npy`, `X_V2.npy` | 203 MB x 2 | ya | tidak — dibuat ulang 32 detik |
| `data/metadata.csv` | 650 KB | ya | tidak — dibuat ulang otomatis |
| **Hasil ekstraksi fitur (CSV)** | ~3 MB | ya | **ya** |
| Model `.joblib` | ~1 MB | ya | **ya** |

Kesimpulannya: **tidak perlu mount Drive untuk tahap praproses.** Yang mahal itu
mengunduh dan menghitung, dan keduanya cuma butuh beberapa menit. Drive baru
mulai berguna di tahap fitur dan model, karena hasilnya kecil tapi butuh waktu
untuk dibuat.

Mulai tahap fitur, tambahkan di akhir notebook:

```python
from google.colab import drive
drive.mount("/content/drive")

import shutil
shutil.copy("data/fitur_glcm_lbp.csv", "/content/drive/MyDrive/PCVK/")
```

---

## Karena notebook cuma satu, aturan gantian

Satu file berarti tidak ada pembagian otomatis. Kalau dua orang membuka dan
menyimpan bersamaan, yang satu akan menimpa yang lain. Empat aturan:

1. **Satu orang pegang satu waktu.** Bilang di grup sebelum mulai.
2. **Pull sebelum mulai, push begitu selesai.** Jangan tinggalkan notebook
   dalam keadaan sudah diedit tapi belum di-push.
3. **Jangan commit setengah jalan.** Kalau mau berhenti, push dulu walau belum
   rapi.
4. **Soal output:** selama kerja harian, bersihkan output sebelum commit supaya
   diff-nya kecil dan tidak bentrok. Saat mau dikumpulkan atau dinilai, Run all
   lalu simpan lengkap.

Kalau aturan ini dilewat, gejalanya khas: notebook berubah 3000+ baris padahal
isinya cuma "dijalankan ulang", dan `git pull` jadi konflik yang tidak bisa
diselesaikan manual.

---

## Kalau mirror HuggingFace bermasalah

Dataset diunduh dari mirror `rjaditya/sipakmed-dataset`. Kalau mirror itu mati
atau sangat lambat, siapkan sekali dari laptop:

```powershell
cd C:\Users\radit\Project-PCVK
Compress-Archive -Path sipakmed_raw\* -DestinationPath sipakmed_raw.zip
```

Unggah `sipakmed_raw.zip` ke `MyDrive/PCVK/`, lalu di Colab sebelum Run all:

```python
from google.colab import drive
drive.mount("/content/drive")
!unzip -q /content/drive/MyDrive/PCVK/sipakmed_raw.zip -d /content/sipakmed_raw
```

Setelah itu lewati cell persiapan (dataset sudah ada) dan lanjut Run all.

> **Jangan** menyimpan dataset ke Drive sebagai folder berisi 4049 file kecil.
> Drive di Colab lambat untuk banyak file kecil — membaca satu citra bisa
> memakan waktu setengah detik, dan 4049 citra jadi setengah jam. Satu file zip
> jauh lebih cepat karena dibaca berurutan.

---

## Masalah yang sering muncul

| Gejala | Penyebab | Solusi |
|---|---|---|
| `Dataset belum lengkap (0/4049), mengunduh ...` | normal di sesi baru | tunggu, jangan dihentikan |
| Unduhan berhenti di tengah | sesi terputus | jalankan ulang cell persiapan — `snapshot_download` melanjutkan, bukan mengulang dari nol |
| `NameError: name 'preprocess' is not defined` | cell fungsi belum dijalankan | **Run all**, jangan loncat ke cell bawah |
| `ModuleNotFoundError: cv2` | lingkungan Colab direset | `!pip install opencv-python-headless` |
| Hasil hilang saat tab ditutup | belum *Save a copy in GitHub* | ulangi, kali ini simpan |
| Angka fold berbeda dari laporan | dataset beda atau kode berubah | cek `periksa_fold` — splitter-nya deterministik, jadi seharusnya persis sama |
| RAM habis / kernel mati | menyimpan semua varian `.npy` sekaligus | notebook ini hanya menyimpan V0 dan V2 |

---

## Yang perlu diingat soal reproducibility

Angka di laporan harus bisa dihasilkan ulang oleh orang lain. Karena itu:

- Pembagian fold memakai `random_state=42` dan algoritma greedy yang ditulis
  sendiri — hasilnya sama di mesin mana pun, tidak bergantung versi scikit-learn.
- Urutan baris metadata diurutkan deterministik (`label`, `cluster_id`, `sel_ke`),
  sehingga indeks ke-`i` di cache `.npy` selalu merujuk citra yang sama.
- Versi pustaka ikut dicetak di cell kedua. Kalau angka berbeda jauh di mesin
  lain, cek dulu versi OpenCV-nya.
