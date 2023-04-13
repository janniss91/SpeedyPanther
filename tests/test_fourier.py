from speedypanther.synthesis.wave_generation import sample_complex_wave
from speedypanther.analysis.fourier import dft

import numpy as np

def test_dft():
    
    # Create a new complex wave to play around with.
    _, complex_wave = sample_complex_wave(frequencies=[600, 800, 1000, 1600], coeffs=[0.3, 0.2, 0.4, 0.1], end_time=0.1)

    duration = 0.005
    sample_rate = 44_100
    # Take a window of samples out of the complex wave.
    sample_window = complex_wave[:int(duration * sample_rate)]

    # Turn the time representation of the wave into a frequency representation.
    freq_ampls = dft(sample_window)

    expected = np.fft.fft(sample_window)
    assert np.allclose(freq_ampls, expected)
