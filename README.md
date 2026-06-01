# So So! - Project PCV

Sebuah proyek Rhythm Game berbasis Python yang memanfaatkan teknologi Computer Vision. Game ini mendeteksi posisi tangan pemain menggunakan kamera (webcam) secara real-time untuk mengendalikan tameng (shield) guna menghalau notes yang datang dari 4 sisi (`top`, `bottom`, `left`, `right`).

Proyek ini menggunakan kombinasi OpenCV untuk mengambil citra dan menampilkan window, Numpy untuk pengolahan citra dan deteksi warna kulit, serta Pygame Mixer untuk pemutaran audio.

---

## Fitur Utama

- **Deteksi Tangan Real-Time:** Menggunakan segmentasi warna HSV untuk melacak posisi tangan pemain.
- **Stage Loader:** Membaca bagan lagu (chart) berbasis format JSON secara dinamis, mendeteksi ketukan musik secara presisi berdasarkan detik/waktu (timestamp).

---

## Struktur File Proyek

Proyek ini dibangun secara modular dengan pembagian tugas sebagai berikut:

1. **`main.py`**
   *Entry point* utama aplikasi. Berfungsi mengatur *game loop*, inisialisasi kamera, sinkronisasi audio, manajemen menu awal (*Start Menu*), serta menghubungkan modul deteksi tangan dengan logika *rhythm game*.
2. **`hand_detection.py`**
   Modul pengolahan citra. Berfungsi mengonversi *frame* video ke color space HSV, melakukan *skin masking* (segmentasi kulit), mencari komponen terbesar, dan mengekstrak titik tengah koordinat tangan pemain.
3. **`rhythm_game.py`**
   Modul inti logika permainan. Mengatur pergerakan *notes* dari luar ke dalam lingkaran tengah, kalkulasi posisi tameng (*shield*), deteksi tabrakan (*collision detection*), serta penghitungan skor dan kombo.
4. **`stage_loader.py`**
   Modul utilitas untuk memuat konfigurasi panggung/level. Membaca metadata lagu dan susunan koordinat *chart* dari file eksternal berbasis `.json`.
5. **`requirements.txt`**
   Daftar dependensi pustaka (*libraries*) pihak ketiga yang dibutuhkan untuk menjalankan proyek.

---

## Panduan Instalasi & Persiapan

### 1. Prasyarat
Pastikan kamu sudah menginstal **Python 3.10** atau versi yang lebih baru di komputermu.

### 2. Kloning Proyek & Masuk ke Direktori
Clone repository ini.

### 3. Instalasi Dependensi
Instal semua pustaka yang diperlukan menggunakan pip:

```Bash
pip install -r requirements.txt
```
### 4. Struktur Folder Aset Musik/Chart (Rekomendasi)
Pastikan file .json panggung/chart dan file audio musik (.mp3 / .wav) diletakkan pada folder yang sesuai dengan path yang dibaca oleh stage_loader.py.

## Cara Bermain
Jalankan skrip utama:

```Bash
python main.py
```
Menu Utama: Tekan tombol SPASI (Spacebar), tombol S, atau Klik Mouse pada layar untuk memulai permainan dan memutar musik.

Mekanik Game: Gerakkan tanganmu di depan webcam (atau gerakkan mouse jika mode testing aktif) untuk mengarahkan tameng melingkar di tengah layar.

Hadang setiap balok note yang meluncur dari arah luar sesuai irama lagu.

Kontrol Tambahan:

Tekan tombol R untuk mengulang permainan dari awal (Restart).

Tekan tombol Q untuk keluar dari permainan (Quit).

## Konfigurasi & Tuning Pengembang
Kamu dapat menyesuaikan sensitivitas permainan secara langsung di dalam kode:

hand_detection.py: Sesuaikan nilai batas HSV (SKIN_H_MIN, SKIN_S_MIN, dll) jika deteksi kulit kurang optimal karena faktor pencahayaan ruanganmu.

main.py: Ubah variabel TEST_MOUSE_CONTROL = False jika kamu ingin murni bermain menggunakan deteksi kamera tangan tanpa bantuan mouse.