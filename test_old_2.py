import librosa
import numpy as np
import soundfile as sf
from pydub import AudioSegment

def remove_pitches(y, sr, remove_pitch_count):
    """
    Uses an STFT to zero out frequency bins corresponding to pitch classes
    we want to remove. We assign a default removal order so that:
      - remove_pitch_count=0: no removals
      - remove_pitch_count=2: remove pitch classes corresponding to Ab and Bb.
    
    Pitch classes are determined by converting frequency values to a MIDI note:
      midi = 69 + 12 * log2(f/440)
    Then pitch_class = midi % 12.
    
    This function zeros out all time-frequency bins whose estimated pitch class 
    is in our removal set.
    """
    # Define an ordering of pitch classes (0=C, 1=C#/Db, 2=D, ... 11=B).
    # In this default order, the first two entries are 8 and 10,
    # which correspond to G#/Ab and A#/Bb.
    removal_order = [8, 10, 3, 6, 1, 4, 7, 9, 11, 0, 2, 5]
    removal_set = set(removal_order[:remove_pitch_count])
    
    # Compute the short-time Fourier transform (STFT)
    n_fft = 2048
    D = librosa.stft(y, n_fft=n_fft)
    # Compute frequency values (in Hz) for each row/bin in D
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    
    # For each frequency bin, determine its pitch class and zero out bins in our removal_set.
    for i, f in enumerate(freqs):
        # Skip DC or non-positive frequencies
        if f <= 0:
            continue
        # Convert f to a MIDI value (this gives a floating point number)
        midi = 69 + 12 * np.log2(f / 440.0)
        # Round to the nearest MIDI note and get the pitch class (modulo 12)
        midi_int = int(np.round(midi))
        pitch_class = midi_int % 12
        if pitch_class in removal_set:
            D[i, :] = 0  # zero out this frequency bin across all time frames

    # Reconstruct the waveform using inverse STFT
    y_processed = librosa.istft(D)
    return y_processed

def change_audio_properties(audio_path, output_path, tempo_factor, pitch_semitones, remove_pitch_count):
    """
    Loads the audio file, applies time stretching and pitch shifting,
    and then removes (zeros out) the frequency components corresponding to 
    a specified number of pitch classes.
    """
    # Load audio with its native sample rate; librosa loads as a numpy array (y)
    y, sr = librosa.load(audio_path, sr=None)
    
    # Adjust tempo (time-stretch) without affecting pitch
    y_stretched = librosa.effects.time_stretch(y, rate=tempo_factor)
    
    # Adjust pitch (change key) without affecting tempo
    y_shifted = librosa.effects.pitch_shift(y_stretched, sr=sr, n_steps=pitch_semitones)
    
    # Remove specified pitch classes (0 means no removal, 12 would remove everything)
    y_final = remove_pitches(y_shifted, sr, remove_pitch_count)
    
    # Write the processed audio to a temporary WAV file.
    temp_wav = "temp.wav"
    sf.write(temp_wav, y_final, sr)
    
    # Now load the temporary WAV and convert/export it as MP3 using pydub.
    modified_audio = AudioSegment.from_wav(temp_wav)
    modified_audio.export(output_path, format="mp3")

# Example usage:
audio_file = "Abide_With_Me_80_0_0.mp3"
output_file = "output.mp3"
tempo_factor = 1.2       # For example, 20% faster
pitch_semitones = -2      # Raise the key by 2 semitones
remove_pitch_count = 2   # Remove 2 pitch classes (by default: G#/Ab & A#/Bb)

change_audio_properties(audio_file, output_file, tempo_factor, pitch_semitones, remove_pitch_count)