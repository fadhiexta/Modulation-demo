# experiments/run_qpsk_demo.py
"""
QPSK demo (baseband, complex):
- buat bit random (harus genap)
- map 2-bit -> simbol QPSK (I,Q) with Gray-ish mapping
- modulasi baseband (complex), normalisasi agar Eb = 1
- tambahkan AWGN (snr treated as Eb/N0 in dB)
- demodulasi coherent (integrate-and-dump per symbol)
- save plots:
    - time domain (I & Q) TX vs RX (segment)
    - constellation scatter (integrated symbol points)
Usage (from project root):
    python -m experiments.run_qpsk_demo --snr 4 --bit-rate 100 --n-bits 1024
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
from src.utils import ensure_dirs, plot_and_save_signal_segment
from src.modulation import generate_time

def parse_args():
    p = argparse.ArgumentParser(description="QPSK demo (baseband)")
    p.add_argument('--snr', type=float, default=6.0, help='Eb/N0 in dB (per bit)')
    p.add_argument('--bit-rate', type=int, default=100, help='bit rate (bits/sec)')
    p.add_argument('--n-bits', type=int, default=1024, help='number of random bits (make even)')
    p.add_argument('--fs', type=int, default=44100, help='sampling rate (Hz)')
    return p.parse_args()

def qpsk_modulate(bits, bit_rate, fs):
    # ensure even number of bits
    if len(bits) % 2 != 0:
        bits = np.concatenate([bits, [0]])
    k = 2  # bits per symbol
    n_symbols = len(bits) // k
    # mapping: b0,b1 -> I,Q  where I = 1 - 2*b0, Q = 1 - 2*b1 (simple Gray-like)
    bits_reshaped = bits.reshape(n_symbols, k)
    I = 1 - 2*bits_reshaped[:,0]
    Q = 1 - 2*bits_reshaped[:,1]
    symbols = I + 1j*Q  # complex symbols
    # upsample: repeat each symbol samples_per_symbol times
    symbol_rate = bit_rate / k
    samples_per_symbol = int(fs / symbol_rate)
    if samples_per_symbol < 1:
        raise ValueError("Sampling rate too low for given bit_rate.")
    waveform = np.repeat(symbols, samples_per_symbol)
    # normalize so Eb = 1 (derivation => divide by sqrt(samples_per_symbol))
    waveform = waveform / np.sqrt(samples_per_symbol)
    return waveform, samples_per_symbol

def awgn(sig, snr_db, bit_rate, fs):
    """
    AWGN that treats snr_db as Eb/N0 (per bit).
    Works for any modulation as long as waveform normalized so Eb=1 per bit.
    """
    snr_linear = 10 ** (snr_db / 10.0)
    samples_per_bit = int(fs // bit_rate)  # samples per bit (not per symbol)
    noise_power_per_sample = 1.0 / (snr_linear * samples_per_bit)  # Eb=1
    noise = np.sqrt(noise_power_per_sample) * (np.random.randn(len(sig)) + 1j*np.random.randn(len(sig))) / np.sqrt(2)
    # divide by sqrt(2) because complex gaussian composed of two real gaussians each with variance sigma2/2
    return sig + noise

def qpsk_demodulate_baseband(rx, bit_rate, fs, samples_per_symbol):
    # rx is complex baseband waveform
    n_symbols = len(rx) // samples_per_symbol
    rx_cut = rx[:n_symbols * samples_per_symbol]
    rx_reshaped = rx_cut.reshape(n_symbols, samples_per_symbol)
    # integrate (sum) across samples -> decision metric for I and Q
    integrals = rx_reshaped.sum(axis=1)  # complex values
    # decisions: I>0 -> bit0=0, I<0 -> bit0=1 ; Q>0 -> bit1=0 else 1
    I_dec = np.real(integrals)
    Q_dec = np.imag(integrals)
    bit0 = (I_dec < 0).astype(int)
    bit1 = (Q_dec < 0).astype(int)
    bits_hat = np.vstack([bit0, bit1]).T.reshape(-1)
    return bits_hat, integrals

def main():
    args = parse_args()
    ensure_dirs()

    fs = args.fs
    bit_rate = args.bit_rate
    n_bits = args.n_bits
    if n_bits % 2 != 0:
        n_bits += 1

    bits = np.random.randint(0, 2, n_bits)
    tx, samples_per_symbol = qpsk_modulate(bits, bit_rate, fs)
    rx = awgn(tx, args.snr, bit_rate, fs)

    bits_hat, integrals = qpsk_demodulate_baseband(rx, bit_rate, fs, samples_per_symbol)
    bits_hat = bits_hat[:len(bits)]
    ber = np.mean(bits != bits_hat)

    # time axis for plotting short segment
    t = generate_time(len(tx) / fs, fs)

    snr_tag = f"snr{args.snr}".replace('.', 'p')
    tx_fname = f"results/plots/qpsk_tx_snr{snr_tag}.png"
    rx_fname = f"results/plots/qpsk_rx_snr{snr_tag}.png"
    const_fname = f"results/plots/qpsk_constellation_snr{snr_tag}.png"

    # plot TX (real and imag)
    plt.figure(figsize=(9,3))
    plt.plot(t[:200], np.real(tx)[:200], label='TX I (real)', alpha=0.9)
    plt.plot(t[:200], np.imag(tx)[:200], label='TX Q (imag)', alpha=0.7)
    plt.title(f"QPSK TX (segment) - SNR={args.snr} dB")
    plt.xlabel("Time (s)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(tx_fname)
    plt.close()

    # plot RX (real and imag)
    plt.figure(figsize=(9,3))
    plt.plot(t[:200], np.real(rx)[:200], label='RX I (real)', alpha=0.9)
    plt.plot(t[:200], np.imag(rx)[:200], label='RX Q (imag)', alpha=0.7)
    plt.title(f"QPSK RX (segment) - SNR={args.snr} dB")
    plt.xlabel("Time (s)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(rx_fname)
    plt.close()

    # constellation (integrated points)
    plt.figure(figsize=(5,5))
    pts = integrals
    plt.scatter(np.real(pts), np.imag(pts), s=8, alpha=0.6)
    plt.axhline(0, color='k', linewidth=0.6)
    plt.axvline(0, color='k', linewidth=0.6)
    plt.title(f"QPSK Constellation (integrated symbols) - SNR={args.snr} dB\nBER={ber:.6f}")
    plt.xlabel("I")
    plt.ylabel("Q")
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.savefig(const_fname)
    plt.close()

    print(f"QPSK demo done. SNR={args.snr} dB -> BER = {ber:.6f}")
    print(f"Saved: {tx_fname}, {rx_fname}, {const_fname}")

if __name__ == "__main__":
    main()