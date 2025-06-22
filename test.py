import librosa
import numpy as np
import soundfile as sf
from pydub import AudioSegment
import wave
import pyaudio


def generate_shush(num_samples, amp=0.1):
    """
    Generates a 'shush' noise segment.
    This function creates white noise and applies a fade-in and fade-out envelope.
    
    Parameters:
      num_samples: int
         Number of samples for the segment.
      amp: float
         Amplitude scaling for the noise.
    
    Returns:
      A numpy array containing the shush noise.
    """
    # Generate white noise
    noise = np.random.uniform(low=-1.0, high=1.0, size=num_samples) * amp
    
    # Create an envelope: 10% fade-in and 10% fade-out
    fade_len = int(0.1 * num_samples)
    if fade_len < 10:
        fade_len = 10  # ensure a minimum length for the fade
    envelope = np.ones(num_samples)
    envelope[:fade_len] = np.linspace(0, 1, fade_len)
    envelope[-fade_len:] = np.linspace(1, 0, fade_len)
    
    return noise * envelope

def change_audio_properties(audio_path, output_path, tempo_factor, pitch_semitones, remove_pitch_count, use_shush=True):
    """
    Processes an audio file by:
      1. Changing tempo and pitch,
      2. For segments where a specified pitch (or pitch class) is detected, the corresponding segment is replaced
         with a 'shush' noise.
    
    Parameters:
      audio_path: str
         Input audio file path.
      output_path: str
         Output audio file path.
      tempo_factor: float
         Factor for time stretching (e.g., 1.2 is 20% faster).
      pitch_semitones: int
         Semitones by which to shift the pitch.
      remove_pitch_count: int
         An integer 0–12 representing how many pitch classes (from a predetermined ordering) should be replaced.
         (0 leaves the audio unchanged.)
      use_shush: bool
         Whether to replace target segments with shush noise (if False, the audio remains unchanged).
    
    The removal ordering is arbitrary—in this example, if remove_pitch_count is 2, the removal set is {8, 10},
    which might correspond to, for example, Ab and Bb.
    """
    # Load the audio using librosa (gets a numpy array and sample rate)
    y, sr = librosa.load(audio_path, sr=None)
    
    # Apply tempo change
    y_stretched = librosa.effects.time_stretch(y, rate=tempo_factor)
    
    # Apply pitch shift
    y_shifted = librosa.effects.pitch_shift(y_stretched, sr=sr, n_steps=pitch_semitones)
    
    if use_shush and remove_pitch_count > 0:
        # Define removal ordering for pitch classes (0 = C, 1 = C#/Db, …, 11 = B)
        # Here, the first `remove_pitch_count` pitch classes will be targeted
        removal_order = [8, 10, 3, 6, 1, 4, 7, 9, 11, 0, 2, 5]
        removal_set = set(removal_order[:remove_pitch_count])
        
        # Set the hop_length for frame analysis
        hop_length = 512
        
        # Use pyin to estimate the pitch for each frame.
        # Note: pyin works best on monophonic or dominant-pitch signals.
        f0, voiced_flag, voiced_prob = librosa.pyin(
            y_shifted, 
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=sr,
            hop_length=hop_length
        )
        
        # Iterate over each frame. f0 is an array (one entry per hop)
        for i, pitch in enumerate(f0):
            # If no pitch was detected in this frame, skip it.
            if np.isnan(pitch):
                continue
            
            # Convert the detected pitch to a MIDI note, then determine its pitch class.
            midi = int(np.round(69 + 12 * np.log2(pitch / 440.0)))
            pitch_class = midi % 12
            
            if pitch_class in removal_set:
                # Compute the start and end sample indices for this frame.
                start = i * hop_length
                end = min(len(y_shifted), start + hop_length)
                num_samples = end - start
                
                # Generate shush noise and replace this segment.
                y_shifted[start:end] = generate_shush(num_samples, amp=0.1)
    
    # Write the processed audio to a temporary WAV file.
    temp_wav = "temp.wav"
    sf.write(temp_wav, y_shifted, sr)

    # Open the WAV file
    file_path = 'temp.wav'
    wf = wave.open(file_path, 'rb')

    # Initialize PyAudio
    p = pyaudio.PyAudio()

    # Open a stream
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                  channels=wf.getnchannels(),
                  rate=wf.getframerate(),
                  output=True)

    # Read and play the audio in chunks
    chunk = 1024
    data = wf.readframes(chunk)
    while data:
      stream.write(data)
      data = wf.readframes(chunk)

    # Close the stream and PyAudio
    stream.close()
    p.terminate()
    
   #  # Convert the WAV to MP3 using pydub.
   #  modified_audio = AudioSegment.from_wav(temp_wav)
   #  modified_audio.export(output_path, format="mp3")

# Example usage:
audio_file = "Abide_With_Me_80_0_0.mp3"
output_file = "output.mp3"
tempo_factor = 1.5         # 20% faster tempo
pitch_semitones = -2        # Shift pitch up by 2 semitones
remove_pitch_count = 9     # Replace segments with two pitch classes (as defined in removal_order)
use_shush = True           # Replace detected segments with shush noise

change_audio_properties(audio_file, output_file, tempo_factor, pitch_semitones, remove_pitch_count, use_shush)
# import random

# # Generate a list of numbers from 0 to 12
# numbers = list(range(12))

# # Shuffle the list to create a random order
# random.shuffle(numbers)

# print(numbers)