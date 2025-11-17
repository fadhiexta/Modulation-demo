# experiments/run_bpsk_ber.py
"""
BPSK BER vs SNR sweep (treated as Eb/N0).
Run as module from project root, e.g.:
    python -m experiments.run_bpsk_ber --snr-min 0 --snr-max 10 --n-bits 20000 --save-csv
Notes:
 - choose n_bits carefully (large = accurate, slow).
 - for quick runs use n_bits=2000..5000.
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from src.modulation import bpsk_modulate, bpsk_demodulate_baseband
from src.utils import ensure_dirs
from scipy.special import erfc
import time
import csv
import os

def awgn(sig, snr_db, bit_rate, fs):
    """
    Add AWGN such that snr_db is Eb/N0 in dB (SNR per BIT).
    """
    samples_per_bit = int(fs // bit_rate)
    sig_power = np.mean(sig**2)
    Eb = sig_power * samples_per_bit
    snr_linear = 10 ** (snr_db / 10.0)
    noise_power_per_sample = Eb / snr_linear / samples_per_bit
    noise = np.sqrt(noise_power_per_sample) * np.random.randn(len(sig))
    return sig + noise

def bpsk_theory_ber(snr_db):
    snr_linear = 10 ** (snr_db / 10.0)
    return 0.5 * erfc(np.sqrt(snr_linear))

def parse_args():
    p = argparse.ArgumentParser(description="BPSK BER vs SNR Sweep")
    p.add_argument('--snr-min', type=float, default=0.0)
    p.add_argument('--snr-max', type=float, default=10.0)
    p.add_argument('--snr-step', type=float, default=1.0)
    p.add_argument('--n-bits', type=int, default=50000, help='bits per SNR point (reduce for speed)')
    p.add_argument('--bit-rate', type=int, default=100)
    p.add_argument('--fs', type=int, default=44100)
    p.add_argument('--seed', type=int, default=None, help='Optional RNG seed for reproducibility')
    p.add_argument('--save-csv', action='store_true', help='Save numeric results to results/plots/ber_results.csv')
    return p.parse_args()

def simulate_ber_point(snr_db, n_bits, bit_rate, fs):
    bits = np.random.randint(0, 2, n_bits)
    tx = bpsk_modulate(bits, bit_rate, fs)
    rx = awgn(tx, snr_db, bit_rate, fs)
    bits_hat = bpsk_demodulate_baseband(rx, bit_rate, fs)
    bits_hat = bits_hat[:len(bits)]
    ber = np.mean(bits != bits_hat)
    return ber

def main():
    args = parse_args()
    ensure_dirs()

    if args.seed is not None:
        np.random.seed(int(args.seed))
        print(f"[info] RNG seed set to {args.seed}")

    snr_list = np.arange(args.snr_min, args.snr_max + 1e-9, args.snr_step)
    ber_sim = []
    start = time.time()
    total = len(snr_list)
    for i, snr in enumerate(snr_list, start=1):
        print(f"Simulating SNR={snr:.2f} dB ... ({i}/{total})", end=' ')
        ber = simulate_ber_point(snr, args.n_bits, args.bit_rate, args.fs)
        ber_sim.append(ber)
        print(f"BER={ber:.5e}")
    elapsed = time.time() - start
    print("Total sim time: {:.1f} s".format(elapsed))

    ber_sim = np.array(ber_sim)
    ber_theory = bpsk_theory_ber(snr_list)

    # Plot results
    plt.figure(figsize=(8,5))
    plt.semilogy(snr_list, ber_sim, 'o-', label='Simulasi')
    plt.semilogy(snr_list, ber_theory, '--', label='Teori BPSK AWGN')
    plt.xlabel('SNR (dB)')
    plt.ylabel('BER (log scale)')
    plt.grid(True, which='both')
    plt.legend()
    plt.tight_layout()
    out_png = "results/plots/ber_bpsk_snr.png"
    plt.savefig(out_png)
    plt.close()
    print(f"BER sweep finished. Plot saved to {out_png}")

    # Optional: save numeric results to CSV
    if args.save_csv:
        csv_path = "results/plots/ber_results.csv"
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['snr_db', 'ber_sim', 'ber_theory'])
            for s, bs, bt in zip(snr_list, ber_sim, ber_theory):
                writer.writerow([f"{s:.6f}", f"{bs:.12e}", f"{bt:.12e}"])
        print(f"CSV saved to {csv_path}")

if __name__ == "__main__":
    main()
