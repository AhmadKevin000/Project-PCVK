# Menjalankan Proyek di Google Colab

Panduan ini untuk anggota kelompok yang tidak mau (atau tidak bisa) memasang
Python di laptopnya. Semua tahap — unduh dataset, praproses, sampai nanti
ekstraksi fitur — bisa dikerjakan di Colab.

---

## Ringkasan: apa yang hilang dan apa yang disimpan

Colab menghapus seluruh isi `/content` setiap sesi berakhir. Jadi pahami dulu
mana yang perlu dijaga:

| Isi | Ukuran | Hilang saat sesi tutup? | Perlu disimpan ke Drive? |
|---|---|---|---|
| Dataset mentah (4049 BMP) | 562 MB | ya | tidak — unduh ulang ~2 menit |
| Cache `X_V0.npy`, `X_V2.npy` | 203 MB x 2 | ya | tidak — dibuat ulang 32 detik |
| `data/metadata.csv` | 650 KB | ya | tidak — dibuat ulang otomatis |
| **Hasil ekstraksi fitur (CSV)** | ~3 MB | ya | **ya** |
| Model `.joblib` | ~1 MB | ya | **ya** |

Kesimpulannya: **tidak perlu mount Drive untuk tahap praproses.** Yang mahal
itu mengunduh dan menghitung, dan keduanya cuma butuh beberapa menit. Drive
baru mulai berguna di tahap fitur dan model, karena hasilnya kecil tapi butuh
waktu untuk dibuat.

---

## Jalur A — lewat GitHub (utama)

### Sekali saja, dari laptop

Branch harus ada di GitHub dulu. Kalau `git push` belum dilakukan, Colab akan
meng-clone versi lama dan notebooknya gagal.

```bash
cd C:\Users\radit\Project-PCVK
git push origin radita
```

### Setiap kali buka Colab

1. Buka <https://colab.research.google.com>
2. **File > Open notebook > GitHub**, tempel:
   `https://github.com/AhmadKevin000/Project-PCVK/blob/radita/notebooks/02_praproses.ipynb`
3. **Runtime > Run all**

Selesai. Cell nomor 0 otomatis:

- meng-clone repo ke `/content/Project-PCVK`
- pindah ke folder itu
- memasang `huggingface_hub`
- mengunduh dataset kalau belum lengkap

Kalau dijalankan di laptop, cell itu tidak melakukan apa-apa.

### Perkiraan waktu di Colab

| Tahap | Waktu |
|---|---|
| Clone repo | ~5 detik |
| Unduh dataset 562 MB | 1–3 menit |
| Praproses 4049 citra (2 varian) | ~40 detik |
| **Total** | **~3 menit** |

---

## Jalur B — lewat Google Drive (cadangan)

Pakai jalur ini kalau:

- mirror HuggingFace mati atau berubah, atau
- koneksi ke HuggingFace lambat, atau
- repo GitHub belum bisa diakses.

### Sekali saja, dari laptop

Buat satu file zip, lalu unggah ke Drive:

```powershell
cd C:\Users\radit\Project-PCVK
Compress-Archive -Path sipakmed_raw\* -DestinationPath sipakmed_raw.zip
```

Unggah `sipakmed_raw.zip` ke `MyDrive/PCVK/`. Ukurannya sekitar 550 MB — BMP
tidak banyak terkompresi, jadi memang sebesar itu. Ini lambat, tapi sekali
saja, dan jadi asuransi kalau mirror-nya hilang.

### Setiap kali buka Colab

```python
from google.colab import drive
drive.mount("/content/drive")

!git clone -q -b radita https://github.com/AhmadKevin000/Project-PCVK.git /content/Project-PCVK
%cd /content/Project-PCVK
!unzip -q /content/drive/MyDrive/PCVK/sipakmed_raw.zip -d sipakmed_raw
```

Setelah itu **Runtime > Run all**, tapi lewati cell 0 (dataset sudah ada).

> **Jangan** menyimpan dataset ke Drive sebagai folder berisi 4049 file kecil.
> Drive di Colab lambat untuk banyak file kecil — membaca satu citra bisa
> memakan waktu setengah detik, dan 4049 citra jadi setengah jam. Satu file
> zip jauh lebih cepat karena dibaca berurutan.

---

## Menyimpan hasil ke Drive

Mulai tahap fitur, simpan hasilnya. Tambahkan di akhir notebook:

```python
from google.colab import drive
drive.mount("/content/drive")

import shutil
shutil.copy("data/fitur_glcm_lbp.csv", "/content/drive/MyDrive/PCVK/")
```

---

## Masalah yang sering muncul

| Gejala | Penyebab | Solusi |
|---|---|---|
| `git clone` gagal / folder kosong | branch belum di-push, atau repo privat | jalankan `git push origin radita` dari laptop |
| `ModuleNotFoundError: cv2` | lingkungan Colab direset | `!pip install opencv-python-headless` |
| Unduhan berhenti di tengah | sesi terputus | jalankan ulang cell 0 — `snapshot_download` melanjutkan, bukan mengulang dari nol |
| `KeyError: 'kelas'` | notebook versi lama | `git pull` di repo, buka ulang notebooknya |
| Hasil beda dengan laptop | dataset belum lengkap | cek jumlah BMP harus **4049**, bukan 4048 |
| Sesi mati saat praproses | batas waktu Colab gratis | jalankan ulang; prosesnya cuma 40 detik |

---

## Yang perlu diingat soal reproducibility

Angka di laporan harus bisa dihasilkan ulang oleh orang lain. Karena itu:

- Pembagian fold memakai `random_state=42` — hasilnya selalu sama.
- Urutan baris metadata diurutkan deterministik (`label`, `cluster_id`, `sel_ke`),
  sehingga indeks ke-`i` di cache `.npy` selalu merujuk citra yang sama.
- Versi pustaka ikut dicatat di cell pertama. Kalau angka berbeda jauh di
  mesin lain, cek dulu versi OpenCV-nya.
