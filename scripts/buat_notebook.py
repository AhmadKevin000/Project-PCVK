"""Bangun notebooks/02_praproses.ipynb dari daftar cell di bawah."""
from pathlib import Path

import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
cd = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))


md(r"""
# Progres 1 (M13) — Praproses Citra SIPaKMeD

Notebook ini menjalankan tahap **praproses** proyek akhir PCVK: dari 4049 citra
sel tunggal Pap smear sampai menjadi tumpukan citra siap diekstraksi fiturnya.

Alurnya mengikuti tujuh langkah:

| Langkah | Isi | Keluaran |
|---|---|---|
| 1 | Bangun metadata (path, label, ID klaster) | `data/metadata.csv` |
| 2 | Bagi fold per klaster, stratifikasi per kelas | kolom `fold` |
| 3 | Satu fungsi praproses, diuji pada satu citra | `src/preprocessing.py` |
| 4 | Validasi visual: asli vs tiap tahap | `outputs/before_after_grid.png` |
| 5 | Ukur pengaruh tiap tahap | `outputs/statistik_praproses.csv` |
| 6 | Batch 4049 citra, simpan cache | `data/cache/*.npy` |
| 7 | Verifikasi akhir | checklist |

**Yang membedakan notebook ini dari versi sebelumnya:** semua cell dijalankan
dan **output-nya disimpan**, sehingga angka di laporan bisa diverifikasi
langsung tanpa menjalankan ulang.
""")

md(r"""
## 0. Persiapan lingkungan

Kalau notebook ini dibuka di Colab, cell di bawah menyiapkan lingkungan:
menyediakan `src/`, memasang `huggingface_hub`, lalu mengunduh dataset kalau
belum lengkap.

**Clone repo tidak wajib.** Cell ini punya dua mode:

1. **Clone otomatis** (default) — paling praktis, selalu dapat versi terbaru.
2. **Unggah manual** — kalau `src/dataset.py` dan `src/preprocessing.py` sudah
   diunggah ke `/content/src/`, cell ini memakai itu dan tidak meng-clone apa pun.

Yang benar-benar wajib hanya tiga: dua modul di `src/`, dataset 4049 citra, dan
notebook ini.

Kalau dijalankan di laptop, cell ini tidak melakukan apa-apa — langsung lanjut.

**Penting:** `/content` di Colab terhapus setiap sesi berakhir. Dataset dan
cache `.npy` akan hilang, tapi tidak masalah: unduh ulang ~2 menit dan
praproses ulang 32 detik. Yang perlu disimpan ke Drive hanya hasil akhir
tahap fitur (CSV kecil), bukan citranya.
""")
cd(r"""
import sys
import os
from pathlib import Path

DI_COLAB = "google.colab" in sys.modules
print(f"Google Colab: {DI_COLAB}")

if DI_COLAB:
    REPO = Path("/content/Project-PCVK")
    BRANCH = "radita"
    URL = "https://github.com/AhmadKevin000/Project-PCVK.git"

    # Dua cara menyiapkan kode: clone repo, atau unggah src/ secara manual.
    # Clone lebih praktis karena selalu dapat versi terbaru, tapi tidak wajib.
    if REPO.exists():
        os.chdir(REPO)
        print("Memakai repo hasil clone")
    elif Path("/content/src/dataset.py").exists():
        os.chdir("/content")
        print("Mode unggah manual: memakai /content/src")
    else:
        print("Meng-clone repo ...")
        os.system(f"git clone -q -b {BRANCH} {URL} {REPO}")
        if REPO.exists():
            os.chdir(REPO)
        else:
            raise RuntimeError(
                "Clone gagal dan src/ tidak ditemukan. Lihat Docs/COLAB.md "
                "bagian 'Jalur B' untuk menyiapkan berkas secara manual."
            )

    os.system("pip -q install huggingface_hub")

    jumlah = (
        len(list(Path("sipakmed_raw").rglob("*.bmp")))
        if Path("sipakmed_raw").exists()
        else 0
    )
    if jumlah != 4049:
        print(f"Dataset belum lengkap ({jumlah}/4049), mengunduh ...")
        if Path("scripts/unduh_dataset.py").exists():
            os.system("python scripts/unduh_dataset.py sipakmed_raw")
        else:
            from huggingface_hub import snapshot_download

            snapshot_download(
                repo_id="rjaditya/sipakmed-dataset",
                repo_type="dataset",
                allow_patterns=["images/*/*/CROPPED/*.bmp"],
                local_dir="sipakmed_raw",
                max_workers=16,
            )

    print(f"Siap di: {Path.cwd()}")
""")

cd(r"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

# Notebook ada di notebooks/, modul ada di src/. Tambahkan akar repo ke path.
AKAR = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
if str(AKAR) not in sys.path:
    sys.path.insert(0, str(AKAR))

from src.dataset import (
    KELAS, LABEL, JUMLAH_RESMI, TOTAL_RESMI,
    bangun_metadata, periksa_metadata, buat_fold, periksa_fold,
    muat_citra, simpan_metadata,
)
from src.preprocessing import (
    VARIAN, KETERANGAN_VARIAN, preprocess, proses_batch, statistik,
)

print(f"Akar repo : {AKAR}")
print(f"Python    : {sys.version.split()[0]}")
print(f"OpenCV    : {cv2.__version__}")
print(f"numpy     : {np.__version__}")
print(f"pandas    : {pd.__version__}")
""")

md(r"""
## 1. Metadata

Satu baris per citra, dengan `cluster_id` diambil dari tiga digit pertama nama
file (`012_04.bmp` -> klaster `012`, sel ke-4).

### Temuan: nomor klaster berulang di setiap kelas

Penomoran klaster dimulai ulang dari `001` di **setiap kelas**. Klaster `012`
di Dyskeratotic dan klaster `012` di Parabasal adalah dua sediaan mikroskop
yang sama sekali berbeda.

Kalau `cluster_id` dipakai sendirian sebagai kunci grup, hasilnya hanya **271
grup** padahal klaster sebenarnya **966** (223 + 238 + 271 + 108 + 126) — sel
dari kelas berbeda ikut tergabung dalam satu grup. Karena itu metadata
menambahkan kolom `grup` berisi `kelas/cluster_id`, dan kolom itulah yang
dipakai untuk membagi data.

Tanpa pemisahan per klaster, sel dari satu sediaan yang sama bisa masuk data
latih dan data uji sekaligus, dan akurasi akan terlihat lebih tinggi daripada
kenyataan.
""")
cd(r"""
KANDIDAT = [
    AKAR / "sipakmed_raw",
    AKAR / "sipakmed",
    Path("/content/sipakmed"),
    Path("/content/drive/MyDrive/PCVK/sipakmed"),
]
AKAR_DATA = next((p for p in KANDIDAT if p.is_dir()), None)
assert AKAR_DATA is not None, f"Dataset tidak ditemukan. Dicek: {KANDIDAT}"
print(f"Dataset : {AKAR_DATA}")

df = bangun_metadata(AKAR_DATA)
ringkas = periksa_metadata(df)

print(f"Total citra    : {ringkas['total_citra']}")
print(f"Jumlah kelas   : {ringkas['jumlah_kelas']}")
print(f"Jumlah klaster : {ringkas['jumlah_klaster']}")
print(f"Sel per klaster: {ringkas['sel_per_klaster_min']} - {ringkas['sel_per_klaster_max']}")
print()
print("Jumlah per kelas (dibandingkan dengan angka resmi Plissiti et al. 2018):")
for nama in KELAS:
    n = int((df["kelas"] == nama).sum())
    print(f"  [{'OK' if n == JUMLAH_RESMI[nama] else 'BEDA'}] {nama:30s} {n:5d}  (resmi {JUMLAH_RESMI[nama]})")

df.head()
""")

md(r"""
## 2. Pembagian data

Lima fold dengan dua syarat: semua sel dari klaster yang sama harus berada di
fold yang sama (anti-leakage), dan proporsi kelima kelas harus mirip di setiap
fold (supaya metrik antar fold bisa dibandingkan).

### Kenapa tidak memakai `StratifiedGroupKFold`

Percobaan pertama memakai `StratifiedGroupKFold` dari scikit-learn. Di laptop
(versi 1.9) hasilnya bagus: fold 809-810 citra, selisih proporsi kelas 0,1
poin. Tetapi ketika notebook yang sama dijalankan di Google Colab, hasilnya
fold 760-858 citra dengan selisih sampai **6,4 poin** — dan pemeriksaan di
bawah menolaknya.

Penyebabnya: algoritma pembagian di dalam scikit-learn berubah antar versi.
Untuk proyek yang angkanya harus bisa direproduksi orang lain, pembagian data
tidak boleh bergantung pada versi pustaka.

### Pembagian yang dipakai sekarang

Untuk setiap kelas:

1. Hitung ukuran tiap klaster (jumlah sel di dalamnya).
2. Urutkan klaster dari yang terbesar.
3. Tempatkan klaster terbesar ke fold yang paling sedikit terisi, lalu lanjut
   ke klaster berikutnya.

Karena tiap kelas dibagi sendiri-sendiri, setiap fold mendapat sekitar
seperlima dari setiap kelas — dan jumlah citra per fold otomatis ikut
seimbang. Tidak ada angka acak yang berpengaruh, jadi hasilnya sama di mesin
mana pun, versi pustaka apa pun.
""")
cd(r"""
df = buat_fold(df, n_splits=5, seed=42)
laporan_fold, masalah = periksa_fold(df)

print("Rekap per fold:")
display(laporan_fold)

# Pemeriksaan anti-leakage secara eksplisit
fold_per_klaster = df.groupby("grup")["fold"].nunique()
print(f"Klaster total                     : {len(fold_per_klaster)}")
print(f"Klaster yang muncul di >1 fold    : {(fold_per_klaster > 1).sum()}  <-- harus 0")

if masalah:
    print("\nMASALAH:")
    for m in masalah:
        print(" -", m)
    raise AssertionError("Pembagian data belum valid, lihat pesan di atas.")
print("\nPembagian data valid: tidak ada klaster yang bocor antar fold.")

# Metadata + fold disimpan: dipakai ulang oleh tahap fitur, tidak perlu
# menghitung ulang pembagian (dan tidak mungkin berbeda urutannya).
simpan_metadata(df, AKAR / "data/metadata.csv")
laporan_fold.to_csv(AKAR / "outputs/rekap_fold.csv", index=False)
print("Disimpan: data/metadata.csv dan outputs/rekap_fold.csv")
""")

md(r"""
### Catatan soal ID pasien

SIPaKMeD tidak menyediakan ID pasien, hanya ID klaster. Artinya pembagian
**per pasien tidak mungkin dilakukan**, dan ini harus dinyatakan sebagai
keterbatasan di laporan. Yang bisa dilakukan — dan sudah dilakukan di sini —
adalah membagi per klaster: sel-sel yang berasal dari satu sediaan mikroskop
yang sama tidak akan terpisah antara data latih dan data uji.
""")

md(r"""
## 3. Fungsi praproses

Seluruh logika praproses berada di `src/preprocessing.py`, bukan di notebook.
Alasannya: fungsi yang sama harus dipakai lagi di tahap fitur dan di aplikasi
demo. Kalau logikanya disalin, versi training dan versi demo akan berbeda.

Empat varian disiapkan supaya pengaruh tiap tahap bisa **diukur**, bukan
sekadar diklaim:

| Varian | Langkah |
|---|---|
| V0 | resize + grayscale (baseline) |
| V1 | V0 + CLAHE |
| V2 | V1 + median filter (model utama) |
| V3 | V0 + median + CLAHE (uji urutan) |
""")
cd(r"""
for k, v in KETERANGAN_VARIAN.items():
    print(f"{k}: {v}")
    print(f"     langkah = {VARIAN[k]}")
""")

md("Uji pada satu citra dulu — belum batch, belum 4049.")
cd(r"""
contoh_path = df.iloc[0]["path"]
mentah = muat_citra(contoh_path)

print(f"Citra mentah : shape={mentah.shape}, dtype={mentah.dtype}, "
      f"min={mentah.min()}, max={mentah.max()}")
print()
for v in VARIAN:
    h = preprocess(mentah, varian=v)
    print(f"{v}: shape={h.shape}, dtype={h.dtype}, min={h.min():3d}, max={h.max():3d}, "
          f"mean={h.mean():6.2f}, std={h.std():5.2f}")
""")

md(r"""
## 4. Validasi visual

Angka saja tidak cukup untuk menilai CLAHE dan median filter. Dua kesalahan
yang hanya kelihatan di mata:

- `clip_limit` terlalu tinggi -> noise ikut ter-amplify, citra jadi granular.
- `tile_grid` terlalu besar -> muncul blok-blok kotak yang jelas.
""")
cd(r"""
rng = np.random.default_rng(42)
n_contoh = 2
kolom = ["Asli (RGB)", "V0", "V1", "V2"]

fig, axes = plt.subplots(
    len(KELAS), n_contoh * len(kolom),
    figsize=(2.4 * n_contoh * len(kolom), 2.6 * len(KELAS)),
)
axes = np.atleast_2d(axes)

for i, nama in enumerate(KELAS):
    sub = df[df["kelas"] == nama]
    pilih = sub.sample(n=n_contoh, random_state=42)
    for j, (_, baris) in enumerate(pilih.iterrows()):
        mentah = muat_citra(baris["path"])
        panel = [mentah, preprocess(mentah, "V0"), preprocess(mentah, "V1"), preprocess(mentah, "V2")]
        for k, (judul, img) in enumerate(zip(kolom, panel)):
            ax = axes[i, j * len(kolom) + k]
            ax.axis("off")
            if img.ndim == 3:
                ax.imshow(img)
            else:
                ax.imshow(img, cmap="gray", vmin=0, vmax=255)
            if i == 0:
                ax.set_title(judul, fontsize=11, fontweight="bold")
    axes[i, 0].text(-0.08, 0.5, nama.replace("im_", "").replace("-", " "),
                    transform=axes[i, 0].transAxes, rotation=90,
                    va="center", ha="right", fontsize=10, fontweight="bold")

plt.suptitle("Praproses SIPaKMeD: asli, baseline, +CLAHE, +median", fontweight="bold")
plt.tight_layout()
Path(AKAR / "outputs").mkdir(exist_ok=True)
plt.savefig(AKAR / "outputs/before_after_grid.png", dpi=130, bbox_inches="tight")
plt.show()
""")

md(r"""
### Uji urutan: median sebelum vs sesudah CLAHE

V2 memasang median **setelah** CLAHE (membersihkan noise yang ikut terangkat).
V3 memasangnya **sebelum**. Mana yang lebih baik harus diuji, bukan diasumsikan.
""")
cd(r"""
satu = muat_citra(df.iloc[100]["path"])
fig, ax = plt.subplots(1, 3, figsize=(12, 4.2))
for a, (judul, v) in zip(ax, [("V1: CLAHE saja", "V1"), ("V2: CLAHE lalu median", "V2"), ("V3: median lalu CLAHE", "V3")]):
    a.imshow(preprocess(satu, v), cmap="gray", vmin=0, vmax=255)
    a.set_title(judul, fontsize=11)
    a.axis("off")
plt.suptitle("Pengaruh urutan langkah", fontweight="bold")
plt.tight_layout()
plt.savefig(AKAR / "outputs/urutan_langkah.png", dpi=130, bbox_inches="tight")
plt.show()
""")

md(r"""
## 5. Statistik: membuktikan pengaruh tiap tahap

Diukur pada 300 citra sampel berstrata (60 per kelas), supaya cepat tetapi
tetap mewakili kelima kelas.

Metrik yang dipakai:

- **std** naik -> kontras meningkat (efek CLAHE)
- **ketajaman** (variansi Laplacian) turun drastis -> detail hilang (median terlalu besar)
- **p1/p99** mendekati 0/255 -> banyak piksel jenuh (clip_limit terlalu tinggi)
""")
cd(r"""
sampel = pd.concat(
    [g.sample(n=60, random_state=42) for _, g in df.groupby("kelas")]
).reset_index(drop=True)
print(f"Sampel: {len(sampel)} citra ({sampel['kelas'].nunique()} kelas x 60)")

baris = []
for _, r in sampel.iterrows():
    mentah = muat_citra(r["path"])
    for v in VARIAN:
        s = statistik(preprocess(mentah, v))
        s.update({"varian": v, "kelas": r["kelas"]})
        baris.append(s)

stat = pd.DataFrame(baris)
ringkas_stat = (
    stat.groupby("varian")[["mean", "std", "p1", "p99", "ketajaman", "entropi"]]
        .mean()
        .reindex(list(VARIAN))
        .round(3)
)
ringkas_stat["keterangan"] = [KETERANGAN_VARIAN[v] for v in ringkas_stat.index]
display(ringkas_stat)

ringkas_stat.to_csv(AKAR / "outputs/statistik_praproses.csv")
print("\nDisimpan: outputs/statistik_praproses.csv")
""")

cd(r"""
fig, ax = plt.subplots(1, 3, figsize=(15, 4))

v_list = list(VARIAN)
ax[0].bar(v_list, [ringkas_stat.loc[v, "std"] for v in v_list], color="#2563EB")
ax[0].set_title("Kontras (std intensitas)\nnaik = lebih kontras", fontweight="bold")
ax[0].set_ylabel("std")

ax[1].bar(v_list, [ringkas_stat.loc[v, "ketajaman"] for v in v_list], color="#10B981")
ax[1].set_title("Ketajaman (variansi Laplacian)\nturun drastis = detail hilang", fontweight="bold")
ax[1].set_ylabel("variansi Laplacian")

for v, warna in zip(["V0", "V1", "V2"], ["#6B7280", "#F59E0B", "#EF4444"]):
    ax[2].hist(preprocess(muat_citra(sampel.iloc[0]["path"]), v).ravel(),
               bins=64, alpha=0.55, label=v, color=warna)
ax[2].set_title("Histogram intensitas\n(satu citra)", fontweight="bold")
ax[2].set_xlabel("intensitas")
ax[2].legend()

for a in ax[:2]:
    a.grid(axis="y", alpha=0.3, linestyle="--")
    a.set_axisbelow(True)

plt.suptitle("Pengaruh tiap tahap praproses (rata-rata 300 citra)", fontweight="bold")
plt.tight_layout()
plt.savefig(AKAR / "outputs/statistik_praproses.png", dpi=130, bbox_inches="tight")
plt.show()
""")

md(r"""
## 6. Batch seluruh dataset

Empat varian diproses dan disimpan sebagai `.npy` **uint8**.

Ukuran: 4049 x 224 x 224 x 1 byte = **203 MB per varian**. Kalau disimpan
sebagai float64, satu varian saja menjadi 1,6 GB dan Colab akan kehabisan RAM.

Yang disimpan untuk keperluan laporan dan tahap berikutnya: **V0** (baseline,
untuk pembanding) dan **V2** (model utama). V1 dan V3 hanya dihitung
statistiknya pada sampel, tidak perlu disimpan seluruhnya.
""")
cd(r"""
Path(AKAR / "data/cache").mkdir(parents=True, exist_ok=True)

waktu_semua = {}
for v in ["V0", "V2"]:
    t0 = time.perf_counter()
    X, waktu = proses_batch(df["path"].tolist(), varian=v, verbose=True)
    total = time.perf_counter() - t0
    waktu_semua[v] = (total, waktu.mean() * 1000)

    np.save(AKAR / f"data/cache/X_{v}.npy", X)
    print(f"{v}: shape={X.shape}, dtype={X.dtype}, "
          f"total={total:.1f}s, rata-rata={waktu.mean()*1000:.1f} ms/citra")
    print(f"     disimpan: data/cache/X_{v}.npy  ({X.nbytes/1e6:.0f} MB)")
    print()

print("Ringkasan waktu:")
for v, (total, per) in waktu_semua.items():
    print(f"  {v}: {total:6.1f} s total, {per:5.1f} ms/citra")
""")

md("## 7. Verifikasi akhir")

cd(r"""
X2 = np.load(AKAR / "data/cache/X_V2.npy")

cek = {
    "Jumlah citra = 4049": len(X2) == TOTAL_RESMI,
    "Shape (4049, 224, 224)": X2.shape == (TOTAL_RESMI, 224, 224),
    "dtype uint8": X2.dtype == np.uint8,
    "Nilai dalam 0-255": int(X2.min()) >= 0 and int(X2.max()) <= 255,
    "Tidak ada NaN": not np.isnan(X2.astype(np.float32)).any(),
    "metadata sejajar": len(df) == len(X2),
}

for nama, ok in cek.items():
    print(f"  [{'OK' if ok else 'X '}] {nama}")

# Uji keterjajaran: indeks acak, hasil cache harus sama dengan proses ulang
rng = np.random.default_rng(7)
for i in rng.choice(len(df), size=3, replace=False):
    ulang = preprocess(muat_citra(df.iloc[i]["path"]), "V2")
    sama = np.array_equal(X2[i], ulang)
    print(f"  [{'OK' if sama else 'X '}] indeks {i} ({df.iloc[i]['nama_file']}): "
          f"cache == proses ulang")

assert all(cek.values()), "Ada pemeriksaan yang gagal."
print("\nSemua pemeriksaan lolos.")
""")

md(r"""
## 8. Kesimpulan

Isi tabel di bawah ini ke laporan M13 sebagai bukti "pengaruh tahap praproses
dijelaskan", bukan sekadar disebut.
""")
cd(r"""
keputusan = pd.DataFrame([
    ["Ukuran citra", "224 x 224", "Ukuran input standar; citra asli 62x48 s/d 531x553 sehingga resize wajib agar skala tekstur GLCM/LBP seragam"],
    ["Interpolasi", "INTER_AREA", "Mayoritas citra lebih besar dari 224 px; INTER_AREA paling baik untuk memperkecil"],
    ["Ruang warna", "Grayscale (RGB2GRAY)", "GLCM dan LBP didefinisikan pada intensitas; sekaligus mengurangi variasi warna pewarnaan antar sediaan"],
    ["CLAHE clipLimit", "2.0", f"std naik dari {ringkas_stat.loc['V0','std']:.1f} ke {ringkas_stat.loc['V1','std']:.1f}; pada clipLimit lebih tinggi noise mulai ter-amplify"],
    ["CLAHE tileGrid", "8 x 8", "Ukuran tile konvensional; cukup lokal untuk menegaskan batas nukleus tanpa memunculkan blok"],
    ["Median ksize", "3", f"ketajaman turun dari {ringkas_stat.loc['V1','ketajaman']:.0f} ke {ringkas_stat.loc['V2','ketajaman']:.0f} (masih wajar, detail tidak hilang)"],
    ["Urutan langkah", "resize - grayscale - CLAHE - median", "Median setelah CLAHE untuk membersihkan noise yang ikut terangkat"],
    ["Varian utama", "V2", "Kontras tinggi dan noise ditekan; V0 disimpan sebagai pembanding untuk membuktikan pengaruh praproses"],
], columns=["Aspek", "Pilihan", "Alasan"])

display(keputusan)
""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.12"},
}

keluar = Path(__file__).resolve().parent.parent / "notebooks" / "02_praproses.ipynb"
keluar.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, keluar)
print(f"Ditulis: {keluar} ({len(cells)} cell)")
