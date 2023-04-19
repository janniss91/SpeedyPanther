import numpy as np

from speedypanther.analysis.fourier import dft
from speedypanther.synthesis.wave_generation import sample_complex_wave


def test_dft():
    # Create a new complex wave with a typical short term window length of 0.005 s.
    _, sample_window = sample_complex_wave(
        frequencies=[600, 800, 1000, 1600], coeffs=[0.3, 0.2, 0.4, 0.1], end_time=0.005
    )

    # Turn the time representation of the wave into a frequency representation.
    freq_ampls = dft(sample_window)
    expected = np.fft.fft(sample_window)

    assert np.allclose(freq_ampls, expected)
