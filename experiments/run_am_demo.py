# experiments/run_am_demo.py
import numpy as np
from src.modulation import am_modulate, am_demodulate_envelope, generate_time
from src.utils import ensure_dirs, save_audio, plot_and_save_signal_segment

def main():
    ensure_dirs()
    fs = 44100
    duration = 1.0
    fc = 5000
    ka = 1.0

    t = generate_time(duration, fs)
    message = 0.6*np.sin(2*np.pi*440*t) + 0.3*np.sin(2*np.pi*880*t)

    mod = am_modulate(message, fc, fs, ka)
    dem = am_demodulate_envelope(mod)

    save_audio("results/audio/am_mod.wav", mod, fs)
    save_audio("results/audio/am_dem.wav", dem, fs)

    plot_and_save_signal_segment(t, message, "Message", "results/plots/message.png")
    plot_and_save_signal_segment(t, mod, "AM Modulated", "results/plots/am_mod.png")
    plot_and_save_signal_segment(t, dem, "AM Demodulated", "results/plots/am_dem.png")

    print("AM Demo selesai — cek folder results.")

if __name__ == "__main__":
    main()