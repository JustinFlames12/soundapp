from kivy.app import App
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.boxlayout import BoxLayout
from kivy.properties import StringProperty, ListProperty
from kivy.clock import Clock
from kivy.uix.popup import Popup 
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.scrollview import ScrollView
import os
import shutil
from random import randint
import librosa
import numpy as np
# from aubio import pitch
import subprocess

from pydub import AudioSegment
import wave
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
import time
# import psutil


if platform == "android":
    from android.storage import app_storage_path
    from jnius import autoclass, PythonJavaClass, java_method

    print("Changing permissions for FFMPEG")
    import os, stat
    ffmpeg_path = "ffmpeg-android"  # update with your actual path
    os.chmod(ffmpeg_path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)  # equivalent to 0o755
    print("Done with: ffmpeg-android")
    print(os.getcwd())
    ffmpeg_path = "./soundapp/Lib/site-packages/ffmpeg"  # update with your actual path
    os.chmod(ffmpeg_path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)  # equivalent to 0o755
    print("Done with: ffmpeg")


    # Define a Python class that implements the Java OnCompletionListener interface
    class OnCompletionListener(PythonJavaClass):
        __javainterfaces__ = ['android/media/MediaPlayer$OnCompletionListener']
        __javacontext__ = 'app'

        @java_method('(Landroid/media/MediaPlayer;)V')
        def onCompletion(self, mp):
            print("Playback completed!")
            # main_screen = MainScreen(name='main')
            # main_screen.on_song_end()
            Clock.schedule_once(self.run_on_main_thread)

        def run_on_main_thread(self, dt):
            # Assuming you already have a reference to your MainScreen instance
            # You should NOT create a new one here
            app = App.get_running_app()
            main_screen = app.root.get_screen('main')
            main_screen.on_song_end_android()
            print('Playback modification completed!')

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

class WrappedLabel(Label):
    # Based on Tshirtman's answer
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.bind(
            width=lambda *x:
            self.setter('text_size')(self, (self.width, None)),
            texture_size=lambda *x: self.setter('height')(self, self.texture_size[1]))

class SongItem(BoxLayout):
    song_name = StringProperty("")

class RootWidget(BoxLayout):
    pass


class MainScreen(Screen):
    song_selected = False
    view_playlist_condition = True
    media_player = None

    def toggle_view_playlist(self):
        self.ids.view_playlist.disabled = False
        self.ids.playbtn.disabled = False
        # self.ids.pausebtn.disabled = False
        # self.ids.restartbtn.disabled = False
        self.song_selected = False

    def play_random_song(self):
        print(f'Song selected: {self.song_selected}')

        # Disable sliders and dropdown
        self.ids.sliderA.disabled = True
        self.ids.slider1.disabled = True
        self.ids.slider2.disabled = True

        if self.song_selected == False:
            # Reset sound variable
            self.sound = None

            self.ids.playbtn.disabled = True
            self.ids.submitbtn.disabled = False
            # self.ids.pausebtn.disabled = False
            # self.ids.restartbtn.disabled = False
            dropdown_text = self.ids.dropdown.text
            dropdown_text = dropdown_text.replace(' ', '')
            os.chdir('songs')
            os.chdir(dropdown_text)
            # if 'temp.wav' in os.listdir():
            #     os.remove('temp.wav')
            for _ in range(5):
                try:
                    if 'temp.mp3' in os.listdir():
                        os.remove('temp.mp3')
                    elif 'temp.m4a' in os.listdir():
                        os.remove('temp.m4a')
                    # elif 'temp.wav' in os.listdir():
                    #     os.remove('temp.wav')
                    break
                except PermissionError as e:
                    print(f"Permission Error (will try again): {e}")
                    time.sleep(0.5)
            
            list_songs = [song for song in os.listdir() if song != 'temp.wav' or song != 'temp_original.wav']
            random_song = randint(0, len(list_songs) - 1)
            print(f'Song: {list_songs[random_song]}')
            print(f'Tempo: {self.ids.slider1.value}')
            print(f'Pitch: {self.ids.sliderA.value}')
            print(f'Level of difficulty: {self.ids.slider2.value}')

            try:
                # Clock.schedule_once(lambda dt: self.change_audio_properties(list_songs[random_song], f'{list_songs[random_song]}_{self.ids.slider1.value}_{self.ids.sliderA.value}_{self.ids.slider2.value}.mp3', self.ids.slider1.value, self.ids.sliderA.value, self.ids.slider2.value, True), 0.1)
                Clock.schedule_once(lambda dt: self.change_audio_properties(list_songs[random_song], f'{list_songs[random_song]}', self.ids.slider1.value, self.ids.sliderA.value, self.ids.slider2.value), 0.1)
            except Exception as e:
                content = BoxLayout(orientation='vertical')
                popup = Popup(title='ERROR', 
                            content=content)
                scroll_error_1 = ScrollView(size_hint=(1, 0.8))
                scroll_error_1.add_widget(WrappedLabel(text = f"Unable to play song. Please try again.\nError message: {e}", size_hint=(1, None)))
                content.add_widget(scroll_error_1)
                error_btn_1 = Button(text='OK', on_press=popup.dismiss, size_hint_y=0.2)
                content.add_widget(error_btn_1)
                popup.open()

                # Reset after exception occurs
                self.ids.playbtn.disabled = False
                self.ids.submitbtn.disabled = True
                self.ids.sliderA.disabled = False
                self.ids.slider1.disabled = False
                self.ids.slider2.disabled = False

                os.chdir('..')
                os.chdir('..')
            # self.change_audio_properties(list_songs[random_song], f'{list_songs[random_song]}_{self.ids.slider1.value}_{self.ids.sliderA.value}_{self.ids.slider2.value}.mp3', self.ids.slider1.value, self.ids.sliderA.value, 0, True)
            # os.remove("temp.wav")

            # Clock.schedule_once(lambda dt: self.change_audio_properties(os.remove("temp.wav")), 0.1)
            # os.chdir('..')
            # os.chdir('..')
            self.random_song = list_songs[random_song][:-4]
        else:
            if platform == "android":  
                self.media_player.start()
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

    # def pitch_shift(self, y: np.ndarray, semitones: float) -> np.ndarray:
    #     rate = 2 ** (semitones / 12)
    #     y_shifted = self.resample_audio(y, 1 / rate)
    #     return y_shifted
    def pitch_shift(self, input_wav, output_wav, semitone_shift):
        from pysoundtouch import SoundTouch
        # Calculate the pitch factor based on semitones.
        # Shifting by n semitones corresponds to a factor of 2^(n/12)
        pitch_factor = 2 ** (semitone_shift / 12.0)

        # Open the input WAV file.
        with wave.open(input_wav, 'rb') as wf:
            channels = wf.getnchannels()
            sample_rate = wf.getframerate()
            sample_width = wf.getsampwidth()
            n_frames = wf.getnframes()
            audio_frames = wf.readframes(n_frames)

        # Convert the byte data to a NumPy array.
        # This example assumes a 16-bit PCM file.
        audio_data = np.frombuffer(audio_frames, dtype=np.int16)

        # Initialize the SoundTouch object.
        # It processes the audio without changing the overall duration.
        with SoundTouch(sample_rate, channels) as st:
            st.set_pitch(pitch_factor)  # Change pitch without affecting speed.
            st.put_samples(audio_data)
            st.flush()  # Make sure to process any buffered data.

            # Retrieve the processed samples in chunks.
            processed = []
            while True:
                chunk = st.receive_samples(4096)
                if chunk.size == 0:
                    break
                processed.append(chunk)
            if processed:
                processed_audio = np.concatenate(processed)
            else:
                processed_audio = audio_data  # Fallback: no change if nothing retrieved.

        # Write the processed audio to the output file.
        with wave.open(output_wav, 'wb') as wf_out:
            wf_out.setnchannels(channels)
            wf_out.setsampwidth(sample_width)
            wf_out.setframerate(sample_rate)
            wf_out.writeframes(processed_audio.tobytes())


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
    
    def hann_window(self, n):
        """Pure-Python Hann window of length n."""
        return 0.5 - 0.5 * np.cos(2*np.pi * np.arange(n) / (n-1))

    def stft(self, y, n_fft=2048, hop_length=None):
        """
        Compute STFT (complex) of 1D signal y.
        Returns (D, hop_length) where
        D: shape (n_fft, n_frames)
        hop_length: hop size in samples
        """
        hop_length = hop_length or n_fft // 4
        # reflect-pad so windows center at t=0 and t=end
        y = np.pad(y, n_fft//2, mode='reflect')
        n_frames = 1 + (len(y) - n_fft) // hop_length
        # build a strided 2D array of frames
        frames = np.lib.stride_tricks.as_strided(
            y,
            shape=(n_frames, n_fft),
            strides=(hop_length*y.strides[0], y.strides[0])
        )
        # apply window and FFT
        win = self.hann_window(n_fft)
        D = np.fft.rfft(frames * win, axis=1).T
        return D, hop_length

    def istft(self, D, hop_length, length=None):
        """
        Inverse STFT returning real signal.
        D: shape (n_freq_bins, n_frames) complex
        hop_length: hop size used in forward STFT
        length: optional final length to trim/pad to
        """
        n_bins, n_frames = D.shape
        n_fft = (n_bins - 1) * 2
        win = self.hann_window(n_fft)
        # ifft -> time frames
        frames = np.fft.irfft(D.T, axis=1)
        # overlap-add
        y = np.zeros(n_fft + hop_length*(n_frames-1))
        wsum = np.zeros_like(y)
        for i in range(n_frames):
            start = i * hop_length
            y[start:start+n_fft]   += frames[i] * win
            wsum[start:start+n_fft] += win**2
        # normalize by window-squared sum
        nonzero = wsum > 1e-8
        y[nonzero] /= wsum[nonzero]
        # remove padding
        y = y[n_fft//2 : -n_fft//2]
        if length is not None:
            y = y[:length]
        return y

    def phase_vocoder(self, D, rate, hop_length):
        """
        Time-stretch D by factor rate with a basic phase-vocoder.
        rate > 1 → speed up (shorter), rate < 1 → slow down (longer).
        """
        n_bins, n_steps = D.shape
        # new time positions
        time_steps = np.arange(0, n_steps, rate, dtype=np.float64)
        n_stretch = len(time_steps)
        D_stretch = np.zeros((n_bins, n_stretch), dtype=np.complex128)

        # initialize phases
        phase_acc = np.angle(D[:, 0])
        last_phase = phase_acc.copy()
        # expected phase advance per bin per hop
        omega = 2 * np.pi * hop_length * np.arange(n_bins) / n_bins

        for i, t in enumerate(time_steps):
            left = int(np.floor(t))
            right = min(left + 1, n_steps - 1)
            alpha = t - left
            # magnitude interpolation
            mag = (1 - alpha) * np.abs(D[:, left]) + alpha * np.abs(D[:, right])
            # phase difference
            delta = np.angle(D[:, right]) - np.angle(D[:, left])
            delta = delta - 2*np.pi * np.round(delta / (2*np.pi))
            # true phase advance
            delta = delta + omega
            phase_acc += delta
            D_stretch[:, i] = mag * np.exp(1j * phase_acc)

        return D_stretch

    def time_stretch_5(self, y, rate, n_fft=2048, hop_length=None):
        """Stretch audio y by factor rate using pure-NumPy phase vocoder."""
        D, hop = self.stft(y, n_fft=n_fft, hop_length=hop_length)
        D_st = self.phase_vocoder(D, rate, hop)
        return self.istft(D_st, hop)

    def pitch_shift(self, y, sr, n_steps):
        """
        Shift pitch by n_steps semitones (–12..+12)
        without altering the duration.
        """
        # 1) compute semitone ratio
        factor = 2.0 ** (n_steps / 12.0)

        # 2) time-stretch by inverse ratio
        y_st = self.time_stretch_5(y, rate=1.0/factor)

        # 3) resample back to original length
        #    → ensures len(y_shifted) == len(y)
        orig_len = len(y)
        y_shifted = np.interp(
            np.linspace(0, len(y_st)-1, orig_len),
            np.arange(len(y_st)),
            y_st
        ).astype(y.dtype)

        return y_shifted

    def change_audio_properties(
        self, audio_path, output_path,
        tempo_factor=1.0, pitch_semitones=0, remove_pitch_count=0
    ):
        y, sr = sf.read(audio_path)
        if y.ndim > 1:
            y = y.mean(axis=1)

        # Apply pitch shift
        # y = self.pitch_shift(y, pitch_semitones)
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
        chosen_uuid = uuid.uuid4()

        # Change tempo of song if necessary
        if tempo_factor != 1.0:
            shutil.copy('temp.wav', 'temp_original.wav')
            from audiotsm import phasevocoder
            from audiotsm.io.wav import WavReader, WavWriter

            input_path = 'temp_original.wav'
            output_path = "temp.wav"

            with WavReader(input_path) as reader:
                with WavWriter(output_path, reader.channels, reader.samplerate) as writer:
                    tsm = phasevocoder(reader.channels, speed=tempo_factor)
                    tsm.run(reader, writer)


            # Check if the file exists before attempting to remove it
            if os.path.exists(input_path):
                os.remove(input_path)

        # # Change pitch of song if necessary
        # if pitch_semitones != 0.0:
        #     input_path = 'temp_original.wav'
        #     shutil.copy(tmp2, input_path)
        #     self.pitch_shift(input_path, tmp2, semitone_shift=int(pitch_semitones))
        #     # Check if the file exists before attempting to remove it
        #     if os.path.exists(input_path):
        #         os.remove(input_path)

        if pitch_semitones != 0.0:
            shutil.copy('temp.wav', 'temp_original.wav')
            # # # Check if the file exists before attempting to remove it
            # # if os.path.exists("temp.wav"):
            # #     os.remove("temp.wav")
            # # # Build the filter string. For example, for 4 semitones:
            # # filter_str = f"asetrate=44100*2^({int(pitch_semitones)}/12),aresample=44100"
            # # proc = subprocess.Popen([
            # #     "ffmpeg", "-i", "temp_original.wav",
            # #     "-filter:a", filter_str,
            # #     f"temp.wav"
            # # ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            # # # This waits for the process to complete
            # # stdout, stderr = proc.communicate()

            # # shutil.copy(f"temp_{chosen_uuid}.wav", 'temp.wav')
            # # # Check if the file exists before attempting to remove it

            # if platform != "android":
            #     import ffmpeg
            #     # Settings
            #     input_file = 'temp_original.wav'
            #     output_file = 'temp.wav'
            #     semitones = int(pitch_semitones)
            #     sample_rate = 44100

            #     # Construct the filter.
            #     # asetrate changes the pitch by adjusting the rate: multiplying by 2^(semitones/12)
            #     # aresample brings the audio back to the original sample rate.
            #     filter_str = f"asetrate={sample_rate}*2^({semitones}/12),aresample={sample_rate}"

            #     # Construct and run the FFmpeg command using ffmpeg-python
            #     (
            #         ffmpeg
            #         .input(input_file)
            #         .output(output_file, af=filter_str, format='wav')
            #         .overwrite_output()
            #         .run()
            #     )
            # else:
            #     from jnius import autoclass

            #     # Import the FFmpegKit class from the FFmpegKit library.
            #     FFmpegKit = autoclass('com.arthenica.ffmpegkit.FFmpegKit')

            #     command = (
            #         f"-i {input_file}"  # Input file
            #         f"-filter:a \"asetrate=44100*2^({int(pitch_semitones)}/12),aresample=44100\" "  # Filter chain: shift pitch while maintaining duration
            #         f"{output_file}"  # Output file
            #     )

            #     # Execute the command using FFmpegKit
            #     session = FFmpegKit.execute(command)
            #     resultCode = session.getReturnCode().getValue()  # Numeric return code
            #     print("Pitch shift command finished with code:", resultCode)

            wav, sr = sf.read('temp_original.wav')
            out = self.pitch_shift(wav, sr, n_steps=int(pitch_semitones))    # +7 semitones up
            sf.write('temp.wav', out, sr)



            if os.path.exists("temp_original.wav"):
                os.remove("temp_original.wav")


                

        if platform == "android":
            print("Before running convert_wav_to_aac() method")
            self.convert_wav_to_aac(tmp2, "temp.aac")
            print("After running convert_wav_to_aac() method")
            os.rename("temp.aac", "temp.m4a")
            tmp2 = "temp.m4a"
            print(f'os.listdir(): {os.listdir()}')
        else:
            AudioSegment.from_wav(tmp2).export(tmp2, format="mp3")
            print(output_path)

        # self.sound = SoundLoader.load(output_path)
        print(f"tmp2: {tmp2}")

        if platform == "android":
            print("Before calling: play_aac_file")
            self.play_aac_file(tmp2)
            print("After calling: play_aac_file")
        else:      
            print("Before calling: self.sound = SoundLoader.load(tmp2)")
            self.sound = SoundLoader.load(tmp2)
            print("After calling: self.sound = SoundLoader.load(tmp2)")
            self.sound.play()
            self.sound.bind(on_stop=self.on_song_end)


        self.song_selected = True
        self.ids.playbtn.disabled = True
        self.ids.pausebtn.disabled = False
        self.ids.restartbtn.disabled = False
        self.ids.submitbtn.disabled = False



        os.chdir('..')
        os.chdir('..')

    def play_aac_file(self, aac_path):
        from jnius import autoclass, PythonJavaClass, java_method

        MediaPlayer = autoclass('android.media.MediaPlayer')
        File = autoclass('java.io.File')
        FileInputStream = autoclass('java.io.FileInputStream')
        FileDescriptor = autoclass('java.io.FileDescriptor')

        f = File(aac_path)
        fis = FileInputStream(f)
        fd = fis.getFD()

        self.media_player = MediaPlayer()
        # media_player.setDataSource(fd)
        self.media_player.setDataSource(aac_path) 
        self.media_player.prepare()

        # Set the completion listener
        listener = OnCompletionListener()
        self.media_player.setOnCompletionListener(listener)

        self.media_player.start()

        print("Playing AAC audio...")


    def convert_wav_to_aac(self, input_wav_path, output_aac_path):
        from jnius import autoclass, PythonJavaClass, java_method
        import struct


        # Java classes
        MediaCodec = autoclass('android.media.MediaCodec')
        MediaFormat = autoclass('android.media.MediaFormat')
        FileOutputStream = autoclass('java.io.FileOutputStream')
        BufferInfo = autoclass('android.media.MediaCodec$BufferInfo')

        # Read WAV and extract raw PCM
        with open(input_wav_path, 'rb') as f:
            header = f.read(44)  # Skip header
            pcm_data = f.read()

        sample_rate = 44100
        channels = 1
        bit_rate = 128000

        # Prepare encoder
        format = MediaFormat.createAudioFormat("audio/mp4a-latm", sample_rate, channels)
        format.setInteger("bitrate", bit_rate)
        format.setInteger("aac-profile", 2)
        format.setInteger("max-input-size", 16384)

        codec = MediaCodec.createEncoderByType("audio/mp4a-latm")
        codec.configure(format, None, None, MediaCodec.CONFIGURE_FLAG_ENCODE)
        codec.start()

        fos = FileOutputStream(output_aac_path)
        input_buffers = codec.getInputBuffers()
        output_buffers = codec.getOutputBuffers()
        buffer_info = BufferInfo()

        offset = 0
        finished = False

        while not finished:
            in_idx = codec.dequeueInputBuffer(10000)
            if in_idx >= 0:
                buf = input_buffers[in_idx]
                buf.clear()

                remaining = len(pcm_data) - offset
                chunk = min(remaining, buf.capacity())
                buf.put(pcm_data[offset:offset + chunk])
                flags = 0 if chunk > 0 else MediaCodec.BUFFER_FLAG_END_OF_STREAM
                codec.queueInputBuffer(in_idx, 0, chunk, 0, flags)
                offset += chunk
                if chunk == 0:
                    finished = True

            out_idx = codec.dequeueOutputBuffer(buffer_info, 10000)
            while out_idx >= 0:
                out_buf = output_buffers[out_idx]
                out_buf.position(buffer_info.offset)
                out_buf.limit(buffer_info.offset + buffer_info.size)

                raw_frame = bytearray(buffer_info.size)
                out_buf.get(raw_frame)

                # Add ADTS header to each frame
                adts = self.add_adts_header(len(raw_frame), sample_rate, channels)
                full_frame = adts + raw_frame
                fos.write(full_frame)

                codec.releaseOutputBuffer(out_idx, False)
                out_idx = codec.dequeueOutputBuffer(buffer_info, 10000)

        codec.stop()
        codec.release()
        fos.close()
        print(f"AAC file with ADTS headers created: {output_aac_path}")



    def add_adts_header(self, packet_length, sample_rate=44100, channels=1):
        from jnius import autoclass, PythonJavaClass, java_method
        import struct

        freq_index_table = {
            96000: 0, 88200: 1, 64000: 2, 48000: 3,
            44100: 4, 32000: 5, 24000: 6, 22050: 7,
            16000: 8, 12000: 9, 11025: 10, 8000: 11,
            7350: 12
        }
        freq_idx = freq_index_table.get(sample_rate, 4)
        chan_cfg = channels
        profile = 2  # AAC LC
        frame_length = packet_length + 7

        adts = bytearray(7)
        adts[0] = 0xFF
        adts[1] = 0xF9
        adts[2] = ((profile - 1) << 6) + (freq_idx << 2) + (chan_cfg >> 2)
        adts[3] = ((chan_cfg & 3) << 6) + (frame_length >> 11)
        adts[4] = (frame_length >> 3) & 0xFF
        adts[5] = ((frame_length & 7) << 5) + 0x1F
        adts[6] = 0xFC
        return adts


    def on_song_end(self, instance):
        self.ids.playbtn.disabled = False
        self.ids.playbtn.text = "Play Again"
        self.ids.pausebtn.disabled = True
        self.ids.restartbtn.disabled = True

    def on_song_end_android(self):
        self.ids.playbtn.disabled = False
        self.ids.playbtn.text = "Play Again"
        self.ids.pausebtn.disabled = True
        self.ids.restartbtn.disabled = True
        

    def pause_song(self):
        try:
            if platform == "android":
                from jnius import autoclass, PythonJavaClass, java_method
                self.media_player.pause()
            else:
                self.sound.stop()
            self.ids.playbtn.disabled = False
            self.ids.pausebtn.disabled = True
            self.ids.restartbtn.disabled = False
        except Exception as e:
            print(f'There was no playing song to stop. Error Message: {e}')

    def restart_song(self):
        try:
            if platform == "android":
                if self.media_player:
                    self.media_player.seekTo(0)
                    if self.media_player.isPlaying():
                        self.media_player.start()
                        self.ids.playbtn.disabled = True
                        self.ids.pausebtn.disabled = False
                        self.ids.restartbtn.disabled = False
            else:
                self.sound.play()
                self.sound.seek(0)
                self.ids.playbtn.disabled = True
                self.ids.pausebtn.disabled = False
                self.ids.restartbtn.disabled = False
        except Exception as e:
            print(f'An error occured when calling the restart_song() function. Error Message: {e}')

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
        main_screen.ids.playbtn.text = "Play"
        main_screen.ids.submitbtn.disabled = True
        main_screen.ids.sliderA.disabled = False
        main_screen.ids.slider1.disabled = False
        main_screen.ids.slider2.disabled = False


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