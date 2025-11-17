# experiments/run_bpsk_demo.py
"""
BPSK demo:
- buat bit random
- modulasi baseband (0 -> -1, 1 -> +1)
- tambahkan AWGN pada satu nilai SNR contoh (treated as Eb/N0)
- demodulasi dengan integrate-and-dump
- simpan plot TX, RX, dan integrals; tampilkan BER di terminal
Run as module from project root:
    python -m experiments.run_bpsk_demo --snr 6 --bit-rate 100 --n-bits 128
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from src.modulation import bpsk_modulate, bpsk_demodulate_baseband, generate_time
from src.utils import ensure_dirs, plot_and_save_signal_segment
import os

def awgn(sig, snr_db, bit_rate, fs):
    """
    Add AWGN such that snr_db is Eb/N0 (dB). Assumes waveform has Eb=1 per bit.
    """
    samples_per_bit = int(fs // bit_rate)
    snr_linear = 10 ** (snr_db / 10.0)   # Eb/N0 linear
    # noise variance per sample (sigma^2)
    noise_power_per_sample = 1.0 / (snr_linear * samples_per_bit)
    noise = np.sqrt(noise_power_per_sample) * np.random.randn(len(sig))
    return sig + noise

def parse_args():
    p = argparse.ArgumentParser(description="BPSK demo (one SNR)")
    p.add_argument('--snr', type=float, default=6.0, help='Example SNR (Eb/N0) in dB')
    p.add_argument('--bit-rate', type=int, default=100, help='bit rate (bits/sec)')
    p.add_argument('--n-bits', type=int, default=128, help='number of random bits')
    p.add_argument('--fs', type=int, default=44100, help='sampling rate (Hz)')
    return p.parse_args()

def main():
    args = parse_args()
    ensure_dirs()

    fs = args.fs
    bit_rate = args.bit_rate
    n_bits = args.n_bits

    bits = np.random.randint(0, 2, n_bits)
    tx = bpsk_modulate(bits, bit_rate, fs)  # baseband (+1 / -1 repeated)

    # add AWGN at chosen SNR (treated as Eb/N0)
    rx = awgn(tx, args.snr, bit_rate, fs)

    # demodulate
    bits_hat = bpsk_demodulate_baseband(rx, bit_rate, fs)
    bits_hat = bits_hat[:len(bits)]
    ber = np.mean(bits != bits_hat)

    # time axis for plotting (short segment)
    t = generate_time(len(tx) / fs, fs)

    # make SNR-safe tag for filenames (e.g. snr2p5)
    snr_tag = f"snr{args.snr}".replace('.', 'p')

    tx_fname = f"results/plots/bpsk_tx_clean_{snr_tag}.png"
    rx_fname = f"results/plots/bpsk_rx_noisy_{snr_tag}.png"
    int_fname = f"results/plots/bpsk_integrals_{snr_tag}.png"

    # save segment plots (TX, RX)
    plot_and_save_signal_segment(t, tx, "BPSK TX (clean) - segment", tx_fname, n_samples=1000)
    plot_and_save_signal_segment(t, rx, f"BPSK RX (noisy) - SNR={args.snr} dB - segment", rx_fname, n_samples=1000)

    # integrals per bit (decision metric) saved to int_fname
    samples_per_bit = int(fs // bit_rate)
    n_bits_calc = len(rx) // samples_per_bit
    rx_cut = rx[:n_bits_calc * samples_per_bit]
    rx_reshaped = rx_cut.reshape(n_bits_calc, samples_per_bit)
    integrals = rx_reshaped.sum(axis=1)

    plt.figure(figsize=(8,3))
    plt.stem(integrals)
    plt.title(f"Integrals per bit (decision metric) - SNR={args.snr} dB")
    plt.xlabel("Bit index")
    plt.ylabel("Integrated value")
    plt.tight_layout()
    plt.savefig(int_fname)
    plt.close()

    print(f"BPSK demo done. SNR={args.snr} dB -> BER = {ber:.6f}")
    print(f"Saved: {tx_fname}, {rx_fname}, {int_fname}")

if __name__ == "__main__":
    main()
