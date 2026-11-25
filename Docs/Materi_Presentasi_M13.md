# Materi Presentasi M13 — Praproses

**Durasi: ±3 menit bicara, ±5 menit kalau ditambah tanya jawab.** Tujuh slide,
satu pesan per slide.

Kalau pendengar cuma ingat tiga hal, ini yang penting:

1. Dataset dibagi **per klaster**, bukan per sel — supaya tidak ada kebocoran data.
2. Ternyata citra SIPaKMeD **kecil** (median 198 px), jadi 61% justru diperbesar.
3. Median filter **tidak didukung data** — noise cuma 0,78 dari 255.

---

## Slide 1 — Pembuka (30 detik)

**Tampilkan:** judul + "Praproses SIPaKMeD, 4049 citra, 5 kelas"

**Bilang:**

> Kami mengerjakan klasifikasi sel Pap smear ke lima kategori. Presentasi ini
> fokus ke tahap pertama, yaitu praproses. Yang saya sampaikan bukan cuma apa
> yang kami kerjakan, tapi apa yang kami temukan — karena ada dua temuan yang
> mengubah rencana kami.

---

## Slide 2 — Dataset (1 menit)

**Tampilkan:** tabel jumlah per kelas

| Kelas | Jumlah |
|---|---|
| Dyskeratotic | 813 |
| Koilocytotic | 825 |
| Metaplastic | 793 |
| Parabasal | 787 |
| Superficial-Intermediate | 831 |
| **Total** | **4.049** |

**Bilang:**

> Datasetnya SIPaKMeD: 4.049 sel, dari 966 klaster, lima kelas. Kelima angka ini
> kami cocokkan satu per satu dengan paper aslinya, dan semuanya sama. Artinya
> datasetnya utuh, tidak ada berkas yang hilang atau salah kelas.

---

## Slide 3 — Bahaya kebocoran data (1 menit)

**Tampilkan:** satu klaster berisi beberapa sel → dipecah ke fold berbeda = salah

**Bilang:**

> Satu klaster itu berisi 1 sampai 26 sel dari satu sediaan yang sama. Kalau sel
> dari klaster yang sama masuk ke data latih sekaligus data uji, model bisa
> menghafal dan akurasinya jadi palsu. Jadi pembagian kami lakukan per klaster,
> bukan per sel.
>
> Hasilnya lima fold, tiap fold 808 sampai 812 sel, selisih proporsi kelas
> paling besar 0,06 poin persen, dan nol klaster yang bocor.

---

## Slide 4 — Empat varian praproses (1 menit)

**Tampilkan:** tabel V0–V3

| Varian | Isi |
|---|---|
| V0 | resize + grayscale |
| V1 | + CLAHE (kontras lokal) |
| V2 | + median filter |
| V3 | median dulu, baru CLAHE |

**Bilang:**

> Ada empat varian, dari yang paling polos sampai yang paling banyak diolah.
> Tujuannya supaya di tahap berikutnya kami bisa membuktikan mana yang benar-
> benar berpengaruh, bukan cuma mengikuti kebiasaan.
>
> Semuanya lewat satu fungsi yang sama, `preprocess()`. Jadi notebook dan demo
> memakai kode yang persis sama — tidak ada logika yang dobel.

---

## Slide 5 — Temuan 1: citranya kecil (1 menit)

**Tampilkan:** grafik ukuran citra

**Bilang:**

> Temuan pertama. Kami kira citra SIPaKMeD itu besar, jadi akan diperkecil ke
> 224. Ternyata ukuran median sisi terpanjangnya cuma 198 piksel. Artinya 61%
> citra justru **diperbesar**, bukan diperkecil.
>
> Yang paling ekstrem kelas Dyskeratotic: median 118 piksel, hampir semuanya
> diperbesar.
>
> Ini penting, karena memperbesar citra tidak menambah informasi — cuma menebak
> piksel yang tidak ada. Jadi kami catat sebagai keterbatasan, dan di tahap
> berikutnya kami uji ukuran 192 yang lebih dekat ke ukuran asli.

---

## Slide 6 — Temuan 2: median filter tidak didukung data (1,5 menit)

**Tampilkan:** tabel noise

| Tahap | Noise (dari 255) |
|---|---|
| Citra asli | 0,78 |
| setelah resize | 0,42 |
| setelah CLAHE | 1,09 |
| setelah median | 0,74 |

**Bilang:**

> Temuan kedua, dan ini yang paling penting.
>
> Kami memakai median filter untuk membersihkan noise. Setelah diukur, noise
> citra ini cuma 0,78 level dari 255. Setelah CLAHE naik jadi 1,09 — masih nol
> koma. Sementara median memotong energi detail sampai 64%.
>
> Jadi kesimpulan kami: median di sini tidak menyelesaikan masalah yang memang
> ada. Tetap kami jalankan sebagai varian uji, tapi bukan sebagai pilihan utama
> — biar data yang menentukan, bukan asumsi kami.

---

## Slide 7 — Status dan rencana (30 detik)

**Tampilkan:** tiga kotak: selesai / sedang / berikutnya

**Bilang:**

> Statusnya: praproses selesai. 4.049 citra jadi 224 kali 224, waktunya 17
> detik. Berikutnya ekstraksi fitur GLCM dan LBP, lalu model dan evaluasi.
>
> Kalau ada yang melihat kelemahan di cara kami mengukur, saya terbuka
> sekarang.

---

## Kalau ditanya

| Pertanyaan | Jawaban singkat |
|---|---|
| Kenapa 224×224? | Ukuran umum untuk CNN, kelipatan 32, hemat memori. Tapi kami tahu mediannya 198, jadi 192 kami uji di tahap berikutnya. |
| Kenapa median tetap dipakai kalau tidak berguna? | Supaya bisa dibuktikan, bukan diasumsikan. Tugasnya memang minta bukti pengaruh satu tahap praproses. |
| Kenapa tidak pakai `StratifiedGroupKFold`? | Sudah dicoba, proporsi kelasnya melenceng sampai 6,4 poin. Kami tulis pembagi sendiri yang hasilnya selalu sama. |
| Bagaimana memastikan tidak ada kebocoran? | Dibagi per klaster, lalu diperiksa ulang: nol klaster muncul di dua fold. |
| Kenapa CLAHE? | Menaikkan kontras lokal supaya batas inti dan sitoplasma lebih jelas. Ini yang dipakai paper rujukan. |
| Apa hasil akhirnya nanti? | Bukan diagnosis. Ini klasifikasi untuk penelitian, dievaluasi dengan sensitivitas, spesifisitas, dan F1. |

---

## Jangan bilang ini

Empat kalimat yang **salah** dan gampang terceplos:

| Jangan bilang | Kenapa |
|---|---|
| "Median filter meningkatkan akurasi" | Belum diuji. Baru diuji di M14. |
| "Detail tidak hilang" | Justru turun 64%. Yang benar: tidak terlihat mata, tapi terukur turun. |
| "Datasetnya 5.015 citra" | Salah. 5.015 itu mencampur sel dengan klaster. Yang benar 4.049. |
| "Mayoritas citra diperkecil" | Terbalik. 61% justru diperbesar. |

---

## Cara menjalankan kalau ditanya

Satu notebook, tanpa clone, tanpa upload:

```
https://colab.research.google.com/github/AhmadKevin000/Project-PCVK/blob/radita/notebooks/02_praproses.ipynb
```

Runtime → Run all → 4 menit → selesai. Angkanya selalu sama karena pembagian
fold-nya deterministik.
