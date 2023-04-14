import matplotlib.pyplot as plt
import numpy as np

from typing import Tuple


def plot_waveform(timesteps: np.ndarray, samples: np.ndarray, show_samples=False, title=""):
    plt.xlabel("time (s)")
    plt.ylabel("Amplitude")
    plt.title(title)
    plt.plot(timesteps, samples)
    if show_samples:
        plt.scatter(timesteps, samples)


def plot_impulse_train(timesteps: np.ndarray, samples: np.ndarray):
    amplitude = samples.max()
    plt.ylim([-amplitude - amplitude * 0.1, amplitude + amplitude * 0.1])
    plt.xlabel("time (s)")
    plt.ylabel("Amplitude")
    plt.title("Impulse Train")
    
    plt.scatter(timesteps, samples)


def plot_frequencies(freq_bins: np.ndarray, freq_ampls: np.ndarray, freq_range: Tuple[int] = (0, 22_100), plot_line: bool = False, log: bool = False):
    """
    Plot the (log) magnitudes of the frequency components of a signal.
    
    freq_bins: The frequencies whose amplitudes are stored.
    freq_ampls: The amplitudes for the respective frequencies.
    freq_range: The range of frequencies to plot.
    plot_line: Plot the contour 
    """
    plt.xlim(*freq_range)
    plt.xlabel("Frequency")
    plt.ylabel("Amplitude")
    plt.title("Magnitude Spectrum")
    
    if log:
        freq_ampls = np.log(freq_ampls)
        plt.ylabel("Log Magnitude (decibel)")
        plt.title("Log Magnitude Spectrum")
        
    if plot_line:
        plt.plot(freq_bins, freq_ampls)
        plt.scatter(freq_bins, freq_ampls)
        
    else:
        plt.stem(freq_bins, freq_ampls)