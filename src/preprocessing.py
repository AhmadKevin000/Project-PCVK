"""
preprocessing.py
================
Pipeline praproses citra Pap smear (SIPaKMeD).

Fungsi utama: `preprocess(img, varian="V2")`.

Fungsi ini adalah SATU-SATUNYA tempat praproses didefinisikan. Notebook,
skrip pelatihan, dan aplikasi demo wajib memanggil fungsi ini -- jangan
menyalin logikanya. Kalau disalin, versi training dan versi demo akan
berbeda diam-diam, dan model yang akurasinya 97% di notebook akan
memberi jawaban ngawur saat didemokan.

Varian yang tersedia (lihat VARIAN):
    V0  resize + grayscale                     (baseline)
    V1  V0 + CLAHE
    V2  V1 + median filter                     (model utama)
    V3  V0 + median filter + CLAHE             (uji urutan langkah)
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Definisi varian
# ---------------------------------------------------------------------------
# Urutan langkah ditulis eksplisit supaya bisa dibandingkan secara adil:
# satu varian hanya boleh berbeda SATU langkah dari varian sebelumnya,
# sehingga selisih hasilnya bisa diatribusikan ke langkah itu.

VARIAN: dict[str, tuple[str, ...]] = {
    "V0": ("resize", "grayscale"),
    "V1": ("resize", "grayscale", "clahe"),
    "V2": ("resize", "grayscale", "clahe", "median"),
    "V3": ("resize", "grayscale", "median", "clahe"),
}

KETERANGAN_VARIAN = {
    "V0": "resize + grayscale (baseline)",
    "V1": "V0 + CLAHE",
    "V2": "V1 + median filter (utama)",
    "V3": "V0 + median + CLAHE (uji urutan)",
}


# ---------------------------------------------------------------------------
# Langkah-langkah
# ---------------------------------------------------------------------------

def ke_grayscale(img: np.ndarray) -> np.ndarray:
    """Ubah citra RGB (atau RGBA) menjadi grayscale uint8.

    Input 2 dimensi dianggap sudah grayscale dan dikembalikan apa adanya,
    supaya fungsi ini tetap jalan untuk citra baru yang formatnya lain.
    """
    if img.ndim == 2:
        return img
    if img.ndim != 3:
        raise ValueError(f"Bentuk citra tidak dikenali: {img.shape}")
    if img.shape[2] == 4:
        img = img[:, :, :3]
    if img.shape[2] != 3:
        raise ValueError(f"Jumlah kanal tidak dikenali: {img.shape}")
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def ubah_ukuran(img: np.ndarray, ukuran: int = 224, jaga_aspek: bool = False) -> np.ndarray:
    """Ubah ukuran citra menjadi ukuran x ukuran.

    jaga_aspek=False (default): citra dipaksa persegi. Ini yang dilakukan
        sebagian besar artikel (mis. A3 me-resize ke 227x227), tetapi
        konsekuensinya bentuk sel ikut terdistorsi. Untuk fitur tekstur
        (GLCM/LBP) dampaknya kecil.

    jaga_aspek=True: citra diberi padding sampai persegi lalu di-resize,
        sehingga bentuk sel (rasio nukleus/sitoplasma, circularity) tetap.
        Pakai opsi ini kalau nanti menambahkan fitur bentuk seperti
        Hu moments.

    Interpolasi dipilih otomatis: INTER_AREA saat memperkecil (mayoritas
    citra SIPaKMeD lebih besar dari 224 px), INTER_CUBIC saat memperbesar.
    """
    h, w = img.shape[:2]

    if jaga_aspek:
        sisi = max(h, w)
        atas = (sisi - h) // 2
        bawah = sisi - h - atas
        kiri = (sisi - w) // 2
        kanan = sisi - w - kiri
        img = cv2.copyMakeBorder(
            img, atas, bawah, kiri, kanan, cv2.BORDER_REPLICATE
        )
        h, w = img.shape[:2]

    if h == ukuran and w == ukuran:
        return img

    interp = cv2.INTER_AREA if (h > ukuran or w > ukuran) else cv2.INTER_CUBIC
    return cv2.resize(img, (ukuran, ukuran), interpolation=interp)


def terapkan_clahe(
    img: np.ndarray, clip_limit: float = 2.0, tile_grid: Sequence[int] = (8, 8)
) -> np.ndarray:
    """CLAHE: penyamaan histogram adaptif yang dibatasi.

    Cara kerjanya: citra dibagi menjadi tile (default 8x8 = 64 tile), tiap
    tile disamakan histogramnya sendiri, lalu puncak histogram dipotong di
    clip_limit x rata-rata dan kelebihannya disebar ke bin lain. Pemotongan
    inilah yang membedakan CLAHE dari equalizeHist biasa: tanpa pemotongan,
    noise ikut ter-amplify.

    Syarat: input harus grayscale uint8.
    """
    if img.ndim != 2:
        raise ValueError("CLAHE membutuhkan citra grayscale 1 kanal.")
    if img.dtype != np.uint8:
        raise ValueError("CLAHE membutuhkan citra uint8.")
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=tuple(int(t) for t in tile_grid))
    return clahe.apply(img)


def filter_median(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Filter median: tiap piksel diganti median dari ksize x ksize tetangganya.

    Dipilih daripada Gaussian blur karena median membuang outlier (bintik
    noise, debris) tanpa mengaburkan tepi nukleus -- tepi yang justru jadi
    ciri pembeda utama antar kelas sel.

    Syarat: ksize ganjil dan >= 3 (keterbatasan OpenCV).
    """
    if ksize % 2 == 0 or ksize < 3:
        raise ValueError(f"ksize harus ganjil dan >= 3, diterima: {ksize}")
    if img.dtype != np.uint8:
        raise ValueError("medianBlur membutuhkan citra uint8.")
    return cv2.medianBlur(img, ksize)


# ---------------------------------------------------------------------------
# Fungsi utama
# ---------------------------------------------------------------------------

def preprocess(
    img: np.ndarray,
    varian: str = "V2",
    ukuran: int = 224,
    clip_limit: float = 2.0,
    tile_grid: Sequence[int] = (8, 8),
    ksize: int = 3,
    jaga_aspek: bool = False,
) -> np.ndarray:
    """Jalankan pipeline praproses sesuai varian.

    Parameter
    ---------
    img : array
        Citra mentah. Boleh RGB (H,W,3), RGBA (H,W,4), atau sudah grayscale
        (H,W). Bentuk apa pun diterima selama bisa dikonversi ke grayscale.
    varian : str
        Salah satu kunci VARIAN ("V0".."V3").

    Mengembalikan
    -------------
    np.ndarray uint8 berukuran (ukuran, ukuran).
    """
    if varian not in VARIAN:
        raise KeyError(f"Varian tidak dikenal: {varian}. Pilihan: {list(VARIAN)}")

    x = img
    for langkah in VARIAN[varian]:
        if langkah == "resize":
            x = ubah_ukuran(x, ukuran=ukuran, jaga_aspek=jaga_aspek)
        elif langkah == "grayscale":
            x = ke_grayscale(x)
        elif langkah == "clahe":
            x = terapkan_clahe(x, clip_limit=clip_limit, tile_grid=tile_grid)
        elif langkah == "median":
            x = filter_median(x, ksize=ksize)
        else:  # pragma: no cover
            raise KeyError(f"Langkah tidak dikenal: {langkah}")

    if x.dtype != np.uint8:
        raise RuntimeError(f"Keluaran praproses harus uint8, dapat {x.dtype}")
    return x


def preprocess_path(path: str | Path, **kwargs) -> np.ndarray:
    """Baca citra dari path lalu praproses. Dipakai aplikasi demo."""
    from src.dataset import muat_citra

    return preprocess(muat_citra(path), **kwargs)


# ---------------------------------------------------------------------------
# Batch
# ---------------------------------------------------------------------------

def proses_batch(
    paths: Iterable[str | Path],
    varian: str = "V2",
    verbose: bool = True,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """Praproses banyak citra sekaligus.

    Mengembalikan (tumpukan_citra, waktu_per_citra_detik).
    Hasil disimpan uint8, bukan float: 4049 citra 224x224 uint8 = 203 MB,
    sedangkan float64 = 1,6 GB dan Colab akan kehabisan RAM.
    """
    paths = list(paths)
    n = len(paths)
    hasil = None
    waktu = np.zeros(n, dtype=np.float32)

    import time

    from src.dataset import muat_citra

    for i, p in enumerate(paths):
        t0 = time.perf_counter()
        citra = preprocess(muat_citra(p), varian=varian, **kwargs)
        waktu[i] = time.perf_counter() - t0

        if hasil is None:
            hasil = np.empty((n, *citra.shape), dtype=np.uint8)
        hasil[i] = citra

        if verbose and (i + 1) % 500 == 0:
            print(f"  {i + 1}/{n} citra ...")

    if hasil is None:
        raise ValueError("Daftar path kosong.")
    return hasil, waktu


# ---------------------------------------------------------------------------
# Statistik untuk membuktikan pengaruh praproses
# ---------------------------------------------------------------------------

def entropi(gray: np.ndarray) -> float:
    """Entropi Shannon dari histogram intensitas (bit)."""
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    p = hist / hist.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def statistik(gray: np.ndarray) -> dict[str, float]:
    """Ringkasan kuantitatif satu citra grayscale.

    mean/std   : rata-rata dan sebaran intensitas. std naik = kontras naik.
    p1/p99     : memeriksa clipping (kalau p1=0 dan p99=255, banyak piksel
                 jenuh -- tanda clip_limit CLAHE terlalu tinggi).
    ketajaman  : variansi Laplacian. Turun drastis = citra kehilangan detail
                 (median filter terlalu besar).
    entropi    : kekayaan informasi intensitas.
    """
    g = gray.astype(np.float64)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return {
        "mean": float(g.mean()),
        "std": float(g.std()),
        "min": float(g.min()),
        "max": float(g.max()),
        "p1": float(np.percentile(g, 1)),
        "p99": float(np.percentile(g, 99)),
        "ketajaman": float(lap.var()),
        "entropi": entropi(gray),
    }
