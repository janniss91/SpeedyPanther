from typing import List, Tuple

import numpy as np


def sample_sinewave(
    frequency: int,
    start_time: float = 0.0,
    end_time: float = 1.0,
    amplitude: float = 1.0,
    sample_rate: int = 44_100,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample a sinewave with a specific frequency and duration (sec).
    """
    # The sample period derived from the sample rate.
    sample_period = 1 / sample_rate

    # Create all timesteps that get a sample according to the sample rate.
    timesteps = np.arange(start_time, end_time, sample_period)

    # Create the number of radian values required according to the frequency.
    full_cycle = np.pi * 2
    radian_values = frequency * full_cycle

    # Subtract phase so that sine curve can start at any point.
    phase = start_time * radian_values

    # Create a sample for each timestep according to the frequency.
    samples = np.sin((timesteps * radian_values) - phase) * amplitude

    return timesteps, samples


def sample_complex_wave(
    frequencies: List[int],
    coeffs: List[float],
    start_time: float = 0.0,
    end_time: float = 1.0,
    sample_rate: int = 44_100,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample a number of simple sinve waves, weight them by their coefficients
    and add them together to form a complex wave.
    """
    sample_lists = []
    for frequency, coeff in zip(frequencies, coeffs):
        timesteps, samples = sample_sinewave(
            frequency, start_time=start_time, end_time=end_time, sample_rate=sample_rate
        )
        sample_lists.append(samples * coeff)

    samples = np.stack(sample_lists).sum(axis=0)

    return timesteps, samples


def generate_impulse_train(
    sample_rate: int = 44_100,
    start_time: float = 0.0,
    end_time: float = 1.0,
    amplitude: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    period = 1 / sample_rate
    timesteps = np.arange(start_time, end_time, period)
    samples = np.ones_like(timesteps) * amplitude
    return timesteps, samples


def generate_noise(
    sample_rate: int = 44_100,
    start_time: float = 0.0,
    end_time: float = 1.0,
    amplitude_std: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    period = 1 / sample_rate
    timesteps = np.arange(start_time, end_time, period)
    samples = np.random.randn(*(timesteps.shape))
    return timesteps, samples


def generate_timesteps(
    input_audio: np.ndarray, sample_rate: int = 44_100
) -> np.ndarray:
    """
    Generate the timesteps for an audiosignal according to its length.
    This can be used for plotting in the time domain.
    """
    input_len = input_audio.shape[0]
    end_time = input_len / sample_rate
    return np.linspace(0.0, end_time, input_len)
