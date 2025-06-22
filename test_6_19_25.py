import numpy as np
import soundfile as sf
from pydub import AudioSegment

def generate_shush(n: int, amp: float = 0.1) -> np.ndarray:
    noise = (np.random.rand(n) * 2 - 1) * amp
    fade = max(int(0.1 * n), 10)
    env = np.ones(n)
    env[:fade] = np.linspace(0, 1, fade)
    env[-fade:] = np.linspace(1, 0, fade)
    return noise * env

def resample_audio(y: np.ndarray, factor: float) -> np.ndarray:
    """Simple linear-resample to change length (for tempo/pitch shifts)."""
    x_old = np.linspace(0, 1, len(y))
    x_new = np.linspace(0, 1, int(len(y) * factor))
    return np.interp(x_new, x_old, y)

def pitch_shift(y: np.ndarray, semitones: float) -> np.ndarray:
    rate = 2 ** (semitones / 12)
    y_shifted = resample_audio(y, 1 / rate)
    return y_shifted

def time_stretch(y: np.ndarray, factor: float) -> np.ndarray:
    return resample_audio(y, factor)

def estimate_pitch_class(frame, sr):
    window = np.hanning(len(frame))
    spectrum = np.fft.rfft(frame * window)
    freqs = np.fft.rfftfreq(len(frame), 1/sr)
    idx = np.argmax(np.abs(spectrum))
    f0 = freqs[idx]
    if f0 < 50 or f0 > 5000:
        return None
    midi = int(round(69 + 12 * np.log2(f0 / 440.0)))
    return midi % 12

def process_audio(
    input_path, output_path,
    tempo=1.0, semitones=0, remove_pitch_count=0
):
    y, sr = sf.read(input_path)
    if y.ndim > 1:
        y = y.mean(axis=1)

    # Apply pitch shift
    y = pitch_shift(y, semitones)
    # Apply tempo stretch
    y = time_stretch(y, tempo)

    # Pitch classes to remove
    removal_order = [8, 10, 3, 6, 1, 4, 7, 9, 11, 0, 2, 5]
    removal = set(removal_order[:remove_pitch_count])

    # Frame-by-frame pitch detect & shush
    win, hop = 2048, 512
    out = y.copy()
    for i in range(0, len(y) - win, hop):
        pc = estimate_pitch_class(y[i:i+win], sr)
        if pc is not None and pc in removal:
            out[i:i+hop] = generate_shush(hop)

    temp_wav = "temp.wav"
    sf.write(temp_wav, out, sr)
    AudioSegment.from_wav(temp_wav).export(output_path, format="mp3")

# ───────────── Example ─────────────
process_audio(
    input_path="MHALL.wav",         # supply WAV input
    output_path="output.mp3",
    tempo=1.2,
    semitones=2,
    remove_pitch_count=2
)

ex = [1,2,3]
print(ex[:0.0])