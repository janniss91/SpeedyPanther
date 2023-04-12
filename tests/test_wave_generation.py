import numpy as np

from speedypanther.synthesis.wave_generation import sample_sinewave


def test_sample_sinewave():
    timesteps, samples = sample_sinewave(frequency=2, sample_rate=10)

    expected_timesteps = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    expected_samples = np.array(
        [
            0.00000000e00,
            9.51056516e-01,
            5.87785252e-01,
            -5.87785252e-01,
            -9.51056516e-01,
            -2.44929360e-16,
            9.51056516e-01,
            5.87785252e-01,
            -5.87785252e-01,
            -9.51056516e-01,
        ]
    )

    assert timesteps.shape == expected_timesteps.shape
    assert samples.shape == expected_samples.shape

    assert np.allclose(timesteps, expected_timesteps)
    assert np.allclose(samples, expected_samples)
