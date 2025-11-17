# src/utils.py
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import os

def ensure_dirs():
    os.makedirs("results/plots", exist_ok=True)
    os.makedirs("results/audio", exist_ok=True)

def save_audio(path, data, fs):
    maxv = np.max(np.abs(data))
    if maxv > 0:
        data = data / maxv * 0.9
    sf.write(path, data.astype(np.float32), fs)

def plot_and_save_signal_segment(t, sig, title, filename, n_samples=1000):
    plt.figure(figsize=(10,3))
    plt.plot(t[:n_samples], sig[:n_samples])
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()