"""
dataset.py
==========
Membaca metadata SIPaKMeD dan membagi fold tanpa data leakage.

Modul ini sengaja dipisah dari notebook supaya:
  1. notebook hanya berisi analisis, bukan logika yang dipakai berulang;
  2. tahap fitur (M13) dan model (M14) memakai pembagian fold yang PERSIS sama;
  3. aplikasi demo (M15) memakai loader citra yang sama dengan saat training.

Aturan penting: urutan KELAS dan isi LABEL tidak boleh diubah setelah
eksperimen berjalan. Kalau diubah, model lama akan memprediksi label yang salah
tanpa memunculkan error apa pun.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Konstanta
# ---------------------------------------------------------------------------

# Urutan kelas dikunci di sini (alfabetis, mengikuti nama folder SIPaKMeD).
KELAS = [
    "im_Dyskeratotic",
    "im_Koilocytotic",
    "im_Metaplastic",
    "im_Parabasal",
    "im_Superficial-Intermediate",
]
LABEL = {nama: i for i, nama in enumerate(KELAS)}

# Jumlah resmi menurut Plissiti et al. (2018). Dipakai sebagai PEMBANDING,
# bukan sebagai sumber data. Kalau hasil hitung beda, berarti unduhan bermasalah.
JUMLAH_RESMI = {
    "im_Dyskeratotic": 813,
    "im_Koilocytotic": 825,
    "im_Metaplastic": 793,
    "im_Parabasal": 787,
    "im_Superficial-Intermediate": 831,
}
TOTAL_RESMI = 4049
JUMLAH_KLASTER_RESMI = 966

# Nama file di CROPPED/ berpola NNN_MM.bmp:
#   NNN = nomor citra klaster asal, MM = nomor sel di dalam klaster itu.
POLA_NAMA = re.compile(r"^(\d+)_(\d+)$")


# ---------------------------------------------------------------------------
# 1. Metadata
# ---------------------------------------------------------------------------

def cari_folder_cropped(akar: str | Path, wajib_lengkap: bool = True) -> dict[str, Path]:
    """Temukan folder CROPPED untuk setiap kelas, apa pun susunan foldernya.

    Layout mirror HuggingFace : <akar>/images/<kelas>/<kelas>/CROPPED/
    Layout hasil ekstrak .7z  : <akar>/<kelas>/CROPPED/

    Keduanya ditangani dengan mencari semua folder bernama CROPPED lalu
    mengambil nama folder induknya sebagai nama kelas.

    wajib_lengkap=False berguna saat dataset masih diunduh sebagian.
    """
    akar = Path(akar)
    if not akar.is_dir():
        raise FileNotFoundError(f"Folder dataset tidak ditemukan: {akar}")

    hasil: dict[str, Path] = {}
    for folder in sorted(akar.rglob("CROPPED")):
        kelas = folder.parent.name
        if kelas in LABEL:
            hasil[kelas] = folder

    hilang = set(LABEL) - set(hasil)
    if hilang and wajib_lengkap:
        raise FileNotFoundError(
            f"Folder CROPPED tidak ditemukan untuk kelas: {sorted(hilang)}. "
            f"Periksa apakah dataset sudah diunduh lengkap di {akar}"
        )
    return hasil


def bangun_metadata(akar: str | Path, periksa: bool = True) -> pd.DataFrame:
    """Bangun tabel metadata: satu baris per citra sel tunggal.

    Kolom yang dihasilkan:
        path        path lengkap citra
        kelas       nama kelas asli (im_...)
        label       label integer 0-4
        cluster_id  3 digit pertama nama file
        sel_ke      nomor sel di dalam klaster
        nama_file   nama file tanpa ekstensi
        grup        kelas + cluster_id, kunci grup untuk pembagian data

    Catatan penting soal cluster_id: penomoran klaster dimulai dari 001 di
    SETIAP kelas, jadi "012" di Dyskeratotic dan "012" di Parabasal adalah
    dua sediaan yang berbeda. Karena itu pembagian data memakai kolom `grup`
    (mis. "im_Dyskeratotic/012"), bukan cluster_id sendirian.
    """
    folders = cari_folder_cropped(akar, wajib_lengkap=periksa)
    baris = []
    tak_sesuai = []

    for kelas, folder in folders.items():
        for f in sorted(folder.glob("*.bmp")):
            m = POLA_NAMA.match(f.stem)
            if m is None:
                tak_sesuai.append(str(f))
                continue
            baris.append(
                {
                    "path": str(f),
                    "kelas": kelas,
                    "label": LABEL[kelas],
                    "cluster_id": m.group(1),
                    "sel_ke": int(m.group(2)),
                    "nama_file": f.stem,
                    "grup": f"{kelas}/{m.group(1)}",
                }
            )

    if tak_sesuai:
        raise ValueError(
            f"{len(tak_sesuai)} file tidak mengikuti pola NNN_MM.bmp, "
            f"contoh: {tak_sesuai[:3]}"
        )

    df = pd.DataFrame(baris)
    # Urutan baris dibuat deterministik. Ini penting: file cache .npy harus
    # selalu sejajar dengan urutan baris tabel ini.
    df = df.sort_values(["label", "cluster_id", "sel_ke"], kind="mergesort")
    df = df.reset_index(drop=True)

    if periksa:
        periksa_metadata(df)
    return df


def periksa_metadata(df: pd.DataFrame) -> dict:
    """Pastikan isi metadata sesuai angka resmi. Melempar error kalau tidak."""
    hitung = df["kelas"].value_counts().to_dict()
    beda = {
        k: (JUMLAH_RESMI[k], int(hitung.get(k, 0)))
        for k in KELAS
        if int(hitung.get(k, 0)) != JUMLAH_RESMI[k]
    }
    if beda:
        rinci = ", ".join(f"{k}: resmi {a} vs dihitung {b}" for k, (a, b) in beda.items())
        raise ValueError(f"Jumlah citra per kelas tidak sesuai angka resmi -> {rinci}")
    if len(df) != TOTAL_RESMI:
        raise ValueError(f"Total citra {len(df)}, seharusnya {TOTAL_RESMI}")

    ringkas = {
        "total_citra": len(df),
        "jumlah_kelas": df["kelas"].nunique(),
        "jumlah_klaster": df["grup"].nunique(),
        "sel_per_klaster_min": int(df.groupby("grup").size().min()),
        "sel_per_klaster_max": int(df.groupby("grup").size().max()),
    }
    return ringkas


# ---------------------------------------------------------------------------
# 2. Pembagian data
# ---------------------------------------------------------------------------

def buat_fold(
    df: pd.DataFrame,
    n_splits: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """Bagi data menjadi n_splits fold: klaster utuh, kelas seimbang.

    Kenapa tidak memakai StratifiedGroupKFold dari scikit-learn:

    Pembagiannya bergantung pada versi pustaka. Dengan data yang sama,
    StratifiedGroupKFold di scikit-learn 1.9 menghasilkan fold 809-810 citra
    dengan selisih proporsi kelas 0,1 poin, sedangkan versi yang dipakai
    Google Colab menghasilkan fold 760-858 citra dengan selisih sampai 6,4
    poin. Untuk proyek yang angkanya harus bisa direproduksi orang lain,
    perilaku yang berubah-ubah antar versi seperti ini tidak bisa dipakai.

    Algoritma di sini tidak bergantung versi pustaka sama sekali:

    1. Untuk setiap kelas, hitung ukuran tiap klaster (jumlah sel di dalamnya).
    2. Urutkan klaster dari yang terbesar.
    3. Tempatkan klaster terbesar ke fold yang paling sedikit terisi, lalu
       lanjut ke klaster berikutnya.

    Karena setiap kelas dibagi sendiri-sendiri, tiap fold mendapat sekitar
    1/n_splits dari setiap kelas. Jumlah citra per fold otomatis ikut
    seimbang, sebab proporsi kelima kelas mirip (787-831 citra).

    Ini varian greedy dari algoritma LPT (longest processing time), yang
    selisihnya terbukti kecil untuk masalah penjadwalan seperti ini.

    Parameter `seed` hanya dipakai untuk mengocok urutan klaster yang
    ukurannya sama, supaya fold tidak selalu menerima klaster bernomor kecil.
    Hasilnya tetap sama setiap kali dijalankan.
    """
    df = df.copy()
    peta: dict[str, int] = {}

    for _, sub in df.groupby("kelas", sort=True):
        ukuran = sub.groupby("grup").size()
        ukuran = ukuran.sample(frac=1.0, random_state=seed).sort_values(
            ascending=False, kind="mergesort"
        )

        beban = [0] * n_splits
        for grup, n in ukuran.items():
            # Fold dengan isi paling sedikit; kalau seri, pilih nomor terkecil
            # supaya hasilnya tetap sama setiap kali dijalankan.
            f = min(range(n_splits), key=lambda i: (beban[i], i))
            beban[f] += n
            peta[grup] = f

    df["fold"] = df["grup"].map(peta).astype(int)
    return df


def periksa_fold(df: pd.DataFrame, toleransi_kelas: float = 0.02) -> tuple[pd.DataFrame, list[str]]:
    """Periksa tiga hal yang menentukan valid/tidaknya pembagian data.

    1. Tidak ada klaster yang muncul di lebih dari satu fold (anti-leakage).
    2. Ukuran fold seimbang.
    3. Proporsi tiap kelas di tiap fold dekat dengan proporsi global.
    """
    masalah: list[str] = []

    # (1) Kebocoran klaster antar fold
    fold_per_klaster = df.groupby("grup")["fold"].nunique()
    bocor = fold_per_klaster[fold_per_klaster > 1]
    if len(bocor) > 0:
        masalah.append(
            f"{len(bocor)} klaster muncul di lebih dari satu fold "
            f"(contoh: {list(bocor.index[:5])})"
        )

    # (2) dan (3) Rekap per fold
    proporsi_global = df["kelas"].value_counts(normalize=True)
    baris = []
    for f in sorted(df["fold"].unique()):
        sub = df[df["fold"] == f]
        proporsi = sub["kelas"].value_counts(normalize=True)
        selisih = {
            k: abs(proporsi.get(k, 0.0) - proporsi_global.get(k, 0.0))
            for k in KELAS
        }
        kelas_terburuk = max(selisih, key=lambda k: selisih[k])
        selisih_maks = selisih[kelas_terburuk]
        baris.append(
            {
                "fold": f,
                "n_citra": len(sub),
                "n_klaster": sub["grup"].nunique(),
                "selisih_proporsi_maks": round(float(selisih_maks), 4),
                **{k.replace("im_", ""): int(v) for k, v in sub["kelas"].value_counts().items()},
            }
        )
        if selisih_maks > toleransi_kelas:
            n_dapat = int((sub["kelas"] == kelas_terburuk).sum())
            n_harap = proporsi_global[kelas_terburuk] * len(sub)
            masalah.append(
                f"Fold {f}: kelas {kelas_terburuk.replace('im_', '')} melenceng "
                f"{selisih_maks * 100:.1f} poin "
                f"({n_dapat} citra, seharusnya ~{n_harap:.0f})"
            )

    laporan = pd.DataFrame(baris).sort_values("fold").reset_index(drop=True)

    ukuran = laporan["n_citra"]
    if ukuran.max() > ukuran.mean() * 1.05 or ukuran.min() < ukuran.mean() * 0.95:
        masalah.append(
            f"Ukuran fold tidak seimbang: {ukuran.min()}-{ukuran.max()} "
            f"(rata-rata {ukuran.mean():.0f})"
        )

    return laporan, masalah


# ---------------------------------------------------------------------------
# 3. Membaca citra
# ---------------------------------------------------------------------------

def muat_citra(path: str | Path) -> np.ndarray:
    """Baca citra sebagai array RGB uint8.

    OpenCV membaca BMP sebagai BGR, sedangkan matplotlib/PIL memakai RGB.
    Konversi dilakukan DI SINI SAJA supaya seluruh pipeline konsisten.
    Kalau campur BGR dan RGB, hasil grayscale bergeser tanpa error apa pun.
    """
    import cv2

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"Citra tidak terbaca: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def simpan_metadata(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path
