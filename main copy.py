from kivy.app import App
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.boxlayout import BoxLayout
from kivy.properties import StringProperty, ListProperty
from kivy.clock import Clock
import os
import shutil
from random import randint
import librosa
import numpy as np
# from aubio import pitch
import subprocess

from pydub import AudioSegment
# import wave
# import pyaudio
import random
from difflib import SequenceMatcher
import datetime
import json
import uuid
from kivy.utils import platform
from plyer import filechooser, audio
# import wave
# from audiostream.core import AudioSample
# from ffpyplayer.player import MediaPlayer
from kivy.core.audio import SoundLoader


if platform == "android":
    from android.storage import app_storage_path
    from jnius import autoclass

    # Get app storage path
    app_path = app_storage_path()
    dest_dir = os.path.join(app_path, "_soundfile_data")
    os.makedirs(dest_dir, exist_ok=True)

    # Copy the .so file
    src = os.path.join(os.path.dirname(__file__), "assets", "libsndfile_arm64.so")
    dst = os.path.join(dest_dir, "libsndfile_arm64.so")
    if not os.path.exists(dst):
        shutil.copy(src, dst)

    # Set environment variable so ctypes can find it
    os.environ["LD_LIBRARY_PATH"] = f"{dest_dir}:{os.environ.get('LD_LIBRARY_PATH', '')}"

import soundfile as sf


class SongItem(BoxLayout):
    song_name = StringProperty("")

class RootWidget(BoxLayout):
    pass

class MainScreen(Screen):
    song_selected = False
    view_playlist_condition = True

    def toggle_view_playlist(self):
        self.ids.view_playlist.disabled = False
        self.ids.playbtn.disabled = False
        # self.ids.pausebtn.disabled = False
        # self.ids.restartbtn.disabled = False
        self.song_selected = False

    def play_random_song(self):
        print(f'Song selected: {self.song_selected}')
        if self.song_selected == False:
            self.ids.playbtn.disabled = True
            # self.ids.pausebtn.disabled = False
            # self.ids.restartbtn.disabled = False
            dropdown_text = self.ids.dropdown.text
            dropdown_text = dropdown_text.replace(' ', '')
            os.chdir('songs')
            os.chdir(dropdown_text)
            # if 'temp.wav' in os.listdir():
            #     os.remove('temp.wav')
            if 'temp.mp3' in os.listdir():
                os.remove('temp.mp3')
            list_songs = [song for song in os.listdir()]
            random_song = randint(0, len(list_songs) - 1)
            print(f'Song: {list_songs[random_song]}')
            print(f'Tempo: {self.ids.slider1.value}')
            print(f'Pitch: {self.ids.sliderA.value}')
            print(f'Level of difficulty: {self.ids.slider2.value}')
            # Clock.schedule_once(lambda dt: self.change_audio_properties(list_songs[random_song], f'{list_songs[random_song]}_{self.ids.slider1.value}_{self.ids.sliderA.value}_{self.ids.slider2.value}.mp3', self.ids.slider1.value, self.ids.sliderA.value, self.ids.slider2.value, True), 0.1)
            Clock.schedule_once(lambda dt: self.change_audio_properties(list_songs[random_song], f'{list_songs[random_song]}', self.ids.slider1.value, self.ids.sliderA.value, self.ids.slider2.value), 0.1)
            # self.change_audio_properties(list_songs[random_song], f'{list_songs[random_song]}_{self.ids.slider1.value}_{self.ids.sliderA.value}_{self.ids.slider2.value}.mp3', self.ids.slider1.value, self.ids.sliderA.value, 0, True)
            # os.remove("temp.wav")

            # Clock.schedule_once(lambda dt: self.change_audio_properties(os.remove("temp.wav")), 0.1)
            # os.chdir('..')
            # os.chdir('..')
            self.random_song = list_songs[random_song][:-4]
        else:
            self.sound.play()
            self.ids.playbtn.disabled = True
            self.ids.pausebtn.disabled = False
            self.ids.restartbtn.disabled = False

    def generate_shush(self, n: int, amp: float = 0.1) -> np.ndarray:
        noise = (np.random.rand(n) * 2 - 1) * amp
        fade = max(int(0.1 * n), 10)
        env = np.ones(n)
        env[:fade] = np.linspace(0, 1, fade)
        env[-fade:] = np.linspace(1, 0, fade)
        return noise * env

    def resample_audio(self, y: np.ndarray, factor: float) -> np.ndarray:
        """Simple linear-resample to change length (for tempo/pitch shifts)."""
        x_old = np.linspace(0, 1, len(y))
        x_new = np.linspace(0, 1, int(len(y) * factor))
        return np.interp(x_new, x_old, y)

    def pitch_shift(self, y: np.ndarray, semitones: float) -> np.ndarray:
        rate = 2 ** (semitones / 12)
        y_shifted = self.resample_audio(y, 1 / rate)
        return y_shifted

    def time_stretch(self, y: np.ndarray, factor: float) -> np.ndarray:
        return self.resample_audio(y, factor)

    def estimate_pitch_class(self, frame, sr):
        window = np.hanning(len(frame))
        spectrum = np.fft.rfft(frame * window)
        freqs = np.fft.rfftfreq(len(frame), 1/sr)
        idx = np.argmax(np.abs(spectrum))
        f0 = freqs[idx]
        if f0 < 50 or f0 > 5000:
            return None
        midi = int(round(69 + 12 * np.log2(f0 / 440.0)))
        return midi % 12

    def change_audio_properties(
        self, audio_path, output_path,
        tempo_factor=1.0, pitch_semitones=0, remove_pitch_count=0
    ):
        y, sr = sf.read(audio_path)
        if y.ndim > 1:
            y = y.mean(axis=1)

        # Apply pitch shift
        y = self.pitch_shift(y, pitch_semitones)
        # Apply tempo stretch
        # y = self.time_stretch(y, tempo_factor)

        # Pitch classes to remove
        removal_order = [8, 10, 3, 6, 1, 4, 7, 9, 11, 0, 2, 5]
        print(remove_pitch_count)
        removal = set(removal_order[:int(remove_pitch_count)])

        # Frame-by-frame pitch detect & shush
        win, hop = 2048, 512
        out = y.copy()
        for i in range(0, len(y) - win, hop):
            pc = self.estimate_pitch_class(y[i:i+win], sr)
            if pc is not None and pc in removal:
                out[i:i+hop] = self.generate_shush(hop)

        # temp_wav = "temp.wav"
        # sf.write(temp_wav, out, sr)
        # AudioSegment.from_wav(temp_wav).export(output_path, format="mp3")


        # 6) Write final WAV and convert to MP3
        tmp2 = "temp.wav"
        sf.write(tmp2, out, sr)
        AudioSegment.from_wav(tmp2).export(tmp2, format="mp3")
        print(output_path)

        # self.sound = SoundLoader.load(output_path)
        self.sound = SoundLoader.load(tmp2)
        self.sound.play()
        self.sound.bind(on_stop=self.on_song_end)
        self.song_selected = True
        self.ids.playbtn.disabled = True
        self.ids.pausebtn.disabled = False
        self.ids.restartbtn.disabled = False
    
        os.chdir('..')
        os.chdir('..')
        # print(f".\\audio_files\\{output_path.split('\\')[-1][:-4]}_{tempo_factor}_{pitch_semitones}_{remove_pitch_count}.mp3")
        modified_audio = AudioSegment.from_wav(tmp2)
        backslash_char = "\\"
        audio_files_song = output_path.split(backslash_char)[-1][:-4]
        modified_audio.export(f".\\audio_files\\{audio_files_song}_{tempo_factor}_{pitch_semitones}_{remove_pitch_count}.mp3", format="mp3")



    # def change_audio_properties(self, audio_path, output_path, tempo_factor, pitch_semitones, remove_pitch_count, use_shush=True):
    #     """
    #     Processes an audio file by:
    #     1. Changing tempo and pitch,
    #     2. For segments where a specified pitch (or pitch class) is detected, the corresponding segment is replaced
    #         with a 'shush' noise.
        
    #     Parameters:
    #     audio_path: str
    #         Input audio file path.
    #     output_path: str
    #         Output audio file path.
    #     tempo_factor: float
    #         Factor for time stretching (e.g., 1.2 is 20% faster).
    #     pitch_semitones: int
    #         Semitones by which to shift the pitch.
    #     remove_pitch_count: int
    #         An integer 0–12 representing how many pitch classes (from a predetermined ordering) should be replaced.
    #         (0 leaves the audio unchanged.)
    #     use_shush: bool
    #         Whether to replace target segments with shush noise (if False, the audio remains unchanged).
        
    #     The removal ordering is arbitrary—in this example, if remove_pitch_count is 2, the removal set is {8, 10},
    #     which might correspond to, for example, Ab and Bb.
    #     """
    #     # Load the audio using librosa (gets a numpy array and sample rate)
    #     y, sr = librosa.load(audio_path, sr=None)
        
    #     # Apply tempo change
    #     y_stretched = librosa.effects.time_stretch(y, rate=tempo_factor)
        
    #     # Apply pitch shift
    #     y_shifted = librosa.effects.pitch_shift(y_stretched, sr=sr, n_steps=pitch_semitones)
        
    #     if use_shush and remove_pitch_count > 0:
    #         # Define removal ordering for pitch classes (0 = C, 1 = C#/Db, …, 11 = B)
    #         # Here, the first `remove_pitch_count` pitch classes will be targeted
    #         removal_order = [8, 10, 3, 6, 1, 4, 7, 9, 11, 0, 2, 5]
    #         # Generate a list of numbers from 0 to 12
    #         removal_order = list(range(12))
    #         print(removal_order)
    #         random.shuffle(removal_order)
    #         print(removal_order)
    #         removal_set = set(removal_order[:int(remove_pitch_count)])
            
    #         # Set the hop_length for frame analysis
    #         hop_length = 512
            
    #         # Use pyin to estimate the pitch for each frame.
    #         # Note: pyin works best on monophonic or dominant-pitch signals.
    #         f0, voiced_flag, voiced_prob = librosa.pyin(
    #             y_shifted, 
    #             fmin=librosa.note_to_hz('C2'),
    #             fmax=librosa.note_to_hz('C7'),
    #             sr=sr,
    #             hop_length=hop_length
    #         )
            
    #         # Iterate over each frame. f0 is an array (one entry per hop)
    #         for i, pitch in enumerate(f0):
    #             # If no pitch was detected in this frame, skip it.
    #             if np.isnan(pitch):
    #                 continue
                
    #             # Convert the detected pitch to a MIDI note, then determine its pitch class.
    #             midi = int(np.round(69 + 12 * np.log2(pitch / 440.0)))
    #             pitch_class = midi % 12
                
    #             if pitch_class in removal_set:
    #                 # Compute the start and end sample indices for this frame.
    #                 start = i * hop_length
    #                 end = min(len(y_shifted), start + hop_length)
    #                 num_samples = end - start
                    
    #                 # Generate shush noise and replace this segment.
    #                 y_shifted[start:end] = self.generate_shush(num_samples, amp=0.1)
        
    #     # Write the processed audio to a temporary WAV file.
    #     temp_wav = "temp.wav"
    #     sf.write(temp_wav, y_shifted, sr)

        
    #     # Convert the WAV to MP3 using pydub.
    #     # print(f"{output_path.split('\\')[-1][:-4]}_{tempo_factor}_{pitch_semitones}_{remove_pitch_count}")
    #     modified_audio = AudioSegment.from_wav(temp_wav)
    #     modified_audio.export("temp.mp3", format="mp3")
    #     # self.sound = SoundLoader.load(output_path)
    #     self.sound = SoundLoader.load("temp.mp3")
    #     self.sound.play()
    #     self.sound.bind(on_stop=self.on_song_end)
    #     self.song_selected = True
    #     self.ids.playbtn.disabled = True
    #     self.ids.pausebtn.disabled = False
    #     self.ids.restartbtn.disabled = False

    #     # # self.play_audio()
    #     # print(os.listdir())
    #     # audio.file_path = 'Hot_Cross_Buns.mp3'
    #     # audio.play()

    #     # player = MediaPlayer("temp.wav")
    #     # player.set_volume(1.0)
    #     # player.play()


    #     # # Write the processed audio to a temporary WAV file.
    #     # temp_wav = "temp.wav"
    #     # sf.write(temp_wav, y_shifted, sr)

    #     # # Open the WAV file
    #     # file_path = 'temp.wav'
    #     # wf = wave.open(file_path, 'rb')

    #     # # Read audio parameters
    #     # nchannels = wf.getnchannels()
    #     # sampwidth = wf.getsampwidth()
    #     # framerate = wf.getframerate()
    #     # nframes = wf.getnframes()

    #     # # Read audio data
    #     # data = wf.readframes(nframes)
        
    #     # # Create and play the AudioSample
    #     # sample = AudioSample(data, sample_rate=framerate, sample_size=sampwidth * 8, channels=nchannels)
    #     # sample.play()

    #     # # Initialize PyAudio
    #     # p = pyaudio.PyAudio()

    #     # # Open a stream
    #     # stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
    #     #             channels=wf.getnchannels(),
    #     #             rate=wf.getframerate(),
    #     #             output=True)

    #     # # Read and play the audio in chunks
    #     # chunk = 1024
    #     # data = wf.readframes(chunk)
    #     # while data:
    #     #     stream.write(data)
    #     #     data = wf.readframes(chunk)

    #     # # Close the stream and PyAudio
    #     # stream.close()
    #     # p.terminate()

    #     # Convert the WAV to MP3 using pydub.
    #     # modified_audio = AudioSegment.from_wav(temp_wav)
    #     # modified_audio.export(output_path, format="mp3")
  
    #     os.chdir('..')
    #     os.chdir('..')
    #     # print(f".\\audio_files\\{output_path.split('\\')[-1][:-4]}_{tempo_factor}_{pitch_semitones}_{remove_pitch_count}.mp3")
    #     modified_audio = AudioSegment.from_wav(temp_wav)
    #     backslash_char = "\\"
    #     audio_files_song = output_path.split(backslash_char)[-1][:-4]
    #     modified_audio.export(f".\\audio_files\\{audio_files_song}_{tempo_factor}_{pitch_semitones}_{remove_pitch_count}.mp3", format="mp3")

    # # def play_audio(self):
    # #     audio.file_path = 'temp.wav'
    # #     audio.play()
    #     # self.ids.playbtn.disabled = False
    #     # self.ids.pausebtn.disabled = True
    #     # self.ids.restartbtn.disabled = True

    def on_song_end(self, instance):
        self.ids.playbtn.disabled = False
        self.ids.pausebtn.disabled = True
        self.ids.restartbtn.disabled = True
        

    def pause_song(self):
        try:
            self.sound.stop()
            self.ids.playbtn.disabled = False
            self.ids.pausebtn.disabled = True
            self.ids.restartbtn.disabled = True
        except Exception as e:
            print(f'There was no playing song to stop. Error Message: {e}')
    def build(self):
        return RootWidget()

class PlaylistScreen(Screen):
    # songs = ListProperty(["Song A", "Song B", "Song C"])
    # songs = ListProperty([song[:-4] for song in os.listdir()])

    def on_enter(self, *args):
        # This ensures that every time the screen is displayed, 
        # the RecycleView's data is updated with the current songs.
        # self.ids.playlist_rv.data = [{'song_name': song} for song in self.songs]
        main_screen = self.manager.get_screen("main")
        dropdown_text = main_screen.ids.dropdown.text
        dropdown_text = dropdown_text.replace(' ', '')
        # print(dropdown_text)
        os.chdir('songs')
        os.chdir(dropdown_text)
        # for song in os.listdir():
        #     print(song[:-4])
        self.ids.playlist_rv.data = [{'song_name': song[:-4]} for song in os.listdir() if song != 'temp.wav']
        
        os.chdir('..')
        os.chdir('..')

    def remove_song(self, song):
        main_screen = self.manager.get_screen("main")
        dropdown_text = main_screen.ids.dropdown.text
        dropdown_text = dropdown_text.replace(' ', '')
        os.chdir('songs')
        os.chdir(dropdown_text)
        songs = [song[:-4] for song in os.listdir()]
        # print(songs)
        if song in songs:
            os.remove(song + '.mp3')
            os.chdir('..')
            os.chdir('..')
            self.on_enter()
        else:
            os.chdir('..')
            os.chdir('..')

    def add_song(self):
        # new_song = f"Song {len(self.songs) + 1}"
        # self.songs.append(new_song)
        selected_file = filechooser.open_file(title="Pick a MP3 file..", 
                    filters=[("MP3 Audio File", "*.mp3")])
        print(selected_file[0])
        if selected_file[0][-4:] == '.mp3':
            # self.file_label.text = f"Selected File: {selected_file[0]}"
            # Here, you can add logic to process the file (e.g., upload to a server)
            print(f'Uploading {selected_file[0][:-4]} file')
            main_screen = self.manager.get_screen("main")
            dropdown_text = main_screen.ids.dropdown.text
            dropdown_text = dropdown_text.replace(' ', '')
            os.chdir('songs')
            os.chdir(dropdown_text)
            shutil.copy(selected_file[0], os.getcwd())
            print("File successfully uploaded to playlist.")
            os.chdir('..')
            os.chdir('..')
            self.on_enter()
        else:
            print("Invalid file. Please ensure an MP3 file is selected.")

class UploadScreen(Screen):
    def upload_file(self):
        # Get the selected file
        selected_file = self.ids.filechooser.selection
        # print(selected_file[0][-4:])
        if selected_file[0][-4:] == '.mp3':
            # self.file_label.text = f"Selected File: {selected_file[0]}"
            # Here, you can add logic to process the file (e.g., upload to a server)
            print(f'Uploading {selected_file[0]} file')
            main_screen = self.manager.get_screen("main")
            dropdown_text = main_screen.ids.dropdown.text
            dropdown_text = dropdown_text.replace(' ', '')
            os.chdir('songs')
            os.chdir(dropdown_text)
            shutil.copy(selected_file[0], os.getcwd())
            self.ids.filechooserlabel.text = "File successfully uploaded to playlist."
            os.chdir('..')
            os.chdir('..')
        else:
            self.ids.filechooserlabel.text = "Invalid file. Please ensure an MP3 file is selected."

class ScoreScreen(Screen):
    def getscore(self):
        main_screen = self.manager.get_screen("main")
        guess_input_text = main_screen.ids.guess_input.text
        title_text = main_screen.random_song
        main_screen.song_selected = False

        title_text = title_text.replace('_', ' ')

        # Remove empty spaces at beginning or end of the strings
        guess_input_text_stripped = guess_input_text.strip()
        title_text_stripped = title_text.strip()

        #Show percentage of difference between two strings
        similarity_score = SequenceMatcher(None, guess_input_text_stripped.lower(), title_text_stripped.lower()).ratio() * 100

        # score_screen = self.manager.get_screen("score")
        self.ids.scorelabel.text = f"Song Title: {title_text}\nUser's Guess: {guess_input_text}\nSimilarity Score: {similarity_score}"

        user_log = {'timestamp':datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S"), 'tempo': main_screen.ids.slider1.value, 'key': main_screen.ids.sliderA.value, 
      'level_of_difficulty': main_screen.ids.slider2.value, 
        #   'browser_information': navigator.userAgent,
        #   'user_language': navigator.language, 
      'random_song_chosen': title_text, 
        #   'random_song_chosen_number': randomSongChosenNumber, 
      'user_guess':guess_input_text,
      'user_score': similarity_score}
        print("User log: ")
        print(user_log)

        # Write dictionary to a JSON file
        with open(f".\\data\\{user_log['timestamp']}_{uuid.uuid4()}.json", "w") as json_file:
            json.dump(user_log, json_file, indent=4)  # 'indent' makes the JSON file 

        main_screen.ids.guess_input.text = ""

class MyScreenManager(ScreenManager):
    pass

class GuessThatSongApp(App):
    # Properties to hold the current theme and background color.
    theme = StringProperty('light')  # 'light' or 'dark'
    background_color = ListProperty([0.678, 0.847, 0.902, 1])  # light blue RGBA

    def toggle_theme(self):
        # Toggle between light mode (light blue) and dark mode (dark blue)
        if self.theme == 'light':
            self.theme = 'dark'
            self.background_color = [0, 0, 0.5, 1]  # dark blue RGBA
        else:
            self.theme = 'light'
            self.background_color = [0.678, 0.847, 0.902, 1]
            
    def build(self):
        print(platform)
        return MyScreenManager()



if __name__ == '__main__':
    GuessThatSongApp().run()