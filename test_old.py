import librosa
import numpy as np
import soundfile as sf
from pydub import AudioSegment

def change_audio_properties(audio_path, output_path, tempo_factor, pitch_semitones):
    # Load the audio file and extract raw audio
    audio = AudioSegment.from_file(audio_path)
    
    # Convert to numpy array
    samples = np.array(audio.get_array_of_samples(), dtype=np.float32) / (2**15)  # Normalize audio

    # Load into librosa with correct sample rate
    y, sr = librosa.load(audio_path, sr=None)  # Ensure librosa loads correctly

    # Apply effects
    y_stretched = librosa.effects.time_stretch(y, rate=tempo_factor)  # Change tempo
    y_shifted = librosa.effects.pitch_shift(y_stretched, sr=sr, n_steps=pitch_semitones)  # Change pitch

    # Save the processed audio
    sf.write("temp.wav", y_shifted, sr)

    # Convert back to MP3
    modified_audio = AudioSegment.from_wav("temp.wav")
    modified_audio.export(output_path, format="mp3")

# Example usage:
audio_file = "Abide_With_Me_80_0_0.mp3"
output_file = "output.mp3"
tempo_factor = 1.2  # 20% faster
pitch_semitones = -2  # Raise pitch by 2 semitones

change_audio_properties(audio_file, output_file, tempo_factor, pitch_semitones)