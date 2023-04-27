import glob
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from scipy.io.wavfile import read

TIMIT_PATH = "data/timit"


@dataclass
class TimitObject:
    name: str
    wav: np.ndarray
    sample_rate: int
    num_frames: int
    transcript: str
    # The timings are stored as tuples of (phone_type, start, end).
    word_timings: List[Tuple[str, int, int]]
    phone_timings: List[Tuple[str, int, int]]


def read_wav(path: str, channels: int = 1) -> np.ndarray:
    sample_rate, wav = read(path)
    arr = np.array(wav, dtype=float)

    if len(arr.shape) > 1:
        arr = arr[:, :channels]

    return arr, sample_rate


def read_transcript(path: str) -> Tuple[str, int]:
    with open(path) as f:
        transcript = f.read().strip()
        time = transcript.split()[:2]
        num_frames = int(time[1])
        transcript = transcript.lstrip(" ".join(time)).lstrip()

    return transcript, num_frames


def read_timings(path: str) -> List[Tuple[str, int, int]]:
    """
    This function can read the timings of different segments (phones or words)
    in the TIMIT corpus.
    """
    timings = []
    with open(path) as f:
        for line in f:
            start, end, segment = line.split()
            timings.append((segment, int(start), int(end)))

    return timings


def read_timit_data(
    timit_path: str, file_type: str = "wav", channels: int = 1
) -> Dict[str, np.ndarray]:
    assert file_type in (
        "wav",
        "phn",
        "wrd",
        "txt",
    ), "Invalid file type specified. Choose one out of: (wav, phn, wrd)"

    # Choose the funtion required for the chosen file type.
    read_funcs = {
        "wav": read_wav,
        "phn": read_timings,
        "wrd": read_timings,
        "txt": read_transcript,
    }

    files = {}
    for file_entry in os.listdir(timit_path):
        path = os.path.join(timit_path, file_entry)
        if os.path.isdir(path):
            glob_pattern = os.path.join(path, f"*.{file_type}")
            for f in glob.glob(glob_pattern):
                func = read_funcs[file_type]
                data = func(f)
                name = file_entry + "-" + os.path.basename(f).rstrip(f".{file_type}")

                files[name] = data

    return files


def get_timit_objects(timit_path):
    timit_wavs = read_timit_data(timit_path, "wav")
    timit_phns = read_timit_data(timit_path, "phn")
    timit_wrds = read_timit_data(timit_path, "wrd")
    timit_transcripts = read_timit_data(timit_path, "txt")

    objs = []
    for name, (wav, sample_rate) in timit_wavs.items():
        transcript, num_frames = timit_transcripts[name]
        phone_timings = timit_phns[name]
        word_timings = timit_wrds[name]
        obj = TimitObject(
            name=name,
            wav=wav,
            sample_rate=sample_rate,
            num_frames=num_frames,
            transcript=transcript,
            word_timings=word_timings,
            phone_timings=phone_timings,
        )

        objs.append(obj)
    return objs
