import numpy as np

from speedypanther.synthesis.wave_generation import sample_sinewave


def fourier_analysis(
    sample_window: np.ndarray, duration: float = 0.005, sample_rate: int = 44_100
):
    """
    Fourier Analysis to move a signal from the time domain to the
    frequency domain. This is the simple version of the Fourier Analysis that
    is based on a dot product of the

    sample_window: A window from a signal on which DFT is performed.
    duration: The duration of the sample_window in seconds.
    sample_rate: The number of samples per second.
    """
    # Make sure that the sample window and the sample rate are in line with each other.
    n_samples = sample_window.shape[0]
    assert (
        int(sample_rate * duration) == n_samples
    ), "The sample window and sample rate are not in line with each other."

    min_freq = 1 / duration
    # This is the maximum frequency that can be sampled.
    nyquist_freq = sample_rate / 2
    freq_bins = np.arange(min_freq, nyquist_freq, min_freq)

    freq_ampls = []
    for freq in freq_bins:
        _, freq_samples = sample_sinewave(
            freq, end_time=duration, sample_rate=sample_rate
        )
        # In case of odd numbers, clip the frequency samples to obtain equal vectors.
        freq_samples = freq_samples[:n_samples]
        freq_ampl = sample_window @ freq_samples
        freq_ampls.append(freq_ampl)

    return np.array(freq_bins), np.array(freq_ampls)


def dft(sample_window: np.ndarray):
    n = sample_window.shape[0]
    t = np.arange(0, n)

    out = [(sample_window * np.exp(- 2j * np.pi * t * k / n)).sum() for k in range(n)]

    return np.array(out)
