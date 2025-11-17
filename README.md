Project Python untuk simulasi modulasi AM dan BPSK lengkap dengan:
- Modulasi
- Demodulasi
- Penambahan noise AWGN
- BER (Bit Error Rate)
- Plotting sinyal (message, modulated, noisy, integrals)
- Audio output

Project ini dibuat untuk portofolio Teknik Telekomunikasi.

## Cara menjalankan

1. Aktifkan virtual environment :

   Windows (PowerShell) :
    python -m venv venv
    .\venv\Scripts\Activate.ps1

3. Install pustaka :
    python -m pip install -r requirements.txt

4. Jalankan AM demo :
    python -m experiments.run_am_demo

5. Jalankan BPSK demo :
    python -m experiments.run_bpsk_demo --snr 6 --bit-rate 100 --n-bits 256

6. Jalankan BER sweep :
    python -m experiments.run_bpsk_ber --snr-min 0 --snr-max 10 --snr-step 1 --n-bits 5000

Hasil tersimpan di folder `results/plots` dan `results/audio`.
