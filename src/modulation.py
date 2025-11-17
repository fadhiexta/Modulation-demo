# src/modulation.py
import numpy as np
from scipy import signal

def generate_time(duration_s, fs):
    return np.arange(0, int(duration_s * fs)) / fs

# ---------------------------
# AM
# ---------------------------
def am_modulate(message, fc, fs, ka=1.0):
    t = np.arange(len(message)) / fs
    carrier = np.cos(2 * np.pi * fc * t)
    return (1.0 + ka * message) * carrier

def am_demodulate_envelope(x):
    analytic = signal.hilbert(x)
    envelope = np.abs(analytic)
    return envelope - np.mean(envelope)

# ---------------------------
# BPSK
# ---------------------------
def bpsk_modulate(bits, bit_rate, fs):
    # symbols = +1/-1
    symbols = 2 * bits - 1
    samples_per_bit = int(fs // bit_rate)
    waveform = np.repeat(symbols, samples_per_bit).astype(float)
    # normalize so that energy per bit Eb = 1
    waveform = waveform / (np.sqrt(samples_per_bit))
    return waveform

def bpsk_demodulate_baseband(rx, bit_rate, fs):
    samples_per_bit = int(fs // bit_rate)
    n_bits = len(rx) // samples_per_bit
    rx = rx[:n_bits * samples_per_bit]
    rx_reshaped = rx.reshape(n_bits, samples_per_bit)
    integrals = rx_reshaped.sum(axis=1)
    bit_est = (integrals > 0).astype(int)
    return bit_est