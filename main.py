import os
home_app_dir = os.getcwd()
# home_app_dir = None
os.environ["KIVY_VIDEO"] = "ffpyplayer"

from kivy.app import App
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.boxlayout import BoxLayout
from kivy.properties import StringProperty, ListProperty, BooleanProperty
from kivy.clock import Clock
from kivy.uix.popup import Popup 
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.scrollview import ScrollView
from kivy.uix.textinput import TextInput
from kivy.uix.video import Video
from kivy.uix.image import Image


import shutil
from random import randint
import librosa
import numpy as np
# from aubio import pitch
import subprocess
import wave
# import pyaudio
import random
from difflib import SequenceMatcher
import datetime
import json
import uuid
from kivy.utils import platform

# # Ensure a writable working directory on iOS (app bundle is read-only).
# # Use the user's Documents folder as the app's working dir so the app can
# # create/read 'data', 'songs', and other runtime files without PermissionError.
# if platform == "ios":
#     try:
#         ios_base = os.path.expanduser('~/Documents')
#         os.makedirs(ios_base, exist_ok=True)
#         os.chdir(ios_base)
#         home_app_dir = ios_base
#         print("iOS: changed working directory to", home_app_dir)
#     except Exception as e:
#         print("iOS: failed to change working directory:", e)
#         home_app_dir = os.getcwd()
# else:
#     home_app_dir = os.getcwd()

# Conditionally import pydub (not available on iOS due to subprocess restrictions)
if platform != "ios":
    from pydub import AudioSegment
from plyer import filechooser, audio
# from audiostream.core import AudioSample
# from ffpyplayer.player import MediaPlayer
from kivy.core.audio import SoundLoader
from kivy.uix.progressbar import ProgressBar
import time
# import psutil
import math
import re
import stat
from collections import Counter
from statistics import mode as stat_mode

# Conditionally import pandas (not available on iOS due to C-extension failures)
if platform != "ios":
    import pandas as pd

if platform == "android":
    from android.storage import app_storage_path
    from jnius import autoclass, PythonJavaClass, java_method


    Image.use_pil = True
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

    from os import chmod
    ffprobe_path =  os.path.join(home_app_dir, "assets", "ffprobe")
    # chmod(ffprobe_path, 0o755)
    # Make sure it’s executable
    os.chmod(ffprobe_path, os.stat(ffprobe_path).st_mode | stat.S_IEXEC)

    # Patch pydub to use this version
    from pydub.utils import get_prober_name
    def custom_prober():
        return ffprobe_path
    import pydub.utils
    pydub.utils.get_prober_name = custom_prober
elif platform == "macosx":
    from os import chmod
    # Prefer a bundled ffprobe, but fall back to the system `ffprobe` if the bundled
    # binary is missing or not runnable (e.g. wrong architecture/format).
    ffprobe_asset = os.path.join(home_app_dir, "assets", "ffprobe").replace("\\", "/")
    ffprobe_path = ffprobe_asset
    print(ffprobe_path)
    try:
        # Try to make the asset executable if it exists
        os.chmod(ffprobe_path, os.stat(ffprobe_path).st_mode | stat.S_IEXEC)
        # Verify the bundled binary is runnable by calling `-version`.
        try:
            proc = subprocess.run([ffprobe_path, '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5)
            if proc.returncode != 0:
                raise OSError('bundled ffprobe returned non-zero')
        except Exception:
            print('Bundled ffprobe not runnable; falling back to system ffprobe in PATH')
            ffprobe_path = 'ffprobe'
    except FileNotFoundError:
        print('Bundled ffprobe not found; using system ffprobe')
        ffprobe_path = 'ffprobe'

    # Patch pydub to use the selected ffprobe
    from pydub.utils import get_prober_name
    def custom_prober():
        return ffprobe_path
    import pydub.utils
    pydub.utils.get_prober_name = custom_prober
    
# Conditionally import soundfile (libsndfile/CFFI not available on iOS)
# On iOS we skip importing soundfile to avoid runtime C-extension errors.
if platform != "ios":
    import soundfile as sf
else:
    sf = None

from audiotsm import phasevocoder
from audiotsm.io.wav import WavReader, WavWriter

def extract_frame_number(path):
    # Extracts the number after 'frame_' in something like 'frame_001_delay-0.1s.png'
    match = re.search(r'frame_(\d+)_', os.path.basename(path))
    return int(match.group(1)) if match else -1  # default to -1 if not matched

class AnimatedPopup:
    def __init__(self):
        self.frame_index = 0
        self.anim_delay = 0.25 / 10.0  # 10 FPS

        if platform == "win" or platform == "macosx":
            # FIXED: use forward slashes in paths for Kivy
            self.frames = sorted([
                os.path.join(home_app_dir, 'anim_frames', fname).replace("\\", "/")  # <-- key fix
                for fname in os.listdir('anim_frames')
                if fname.endswith('.png')
            ], key=extract_frame_number)
            # print(self.frames)
        else:
            # Load all valid frame paths
            print(os.getcwd())
            self.frames = sorted([
                os.path.join('anim_frames', fname)
                for fname in os.listdir('anim_frames')
                if fname.endswith('.png')
            ], key=lambda x: int(re.search(r'frame_(\d+)_', x).group(1)))

        # Setup image and popup
        self.image_widget = Image()
        self.image_widget.source = self.frames[0]
        self.image_widget.allow_stretch = True
        self.image_widget.keep_ratio = False

        self.popup = Popup(title="Loading Animation",
                           content=self.image_widget,
                           size_hint=(0.6, 0.6),
                           auto_dismiss=False)

        self.clock_event = None

    def open(self):
        self.popup.open()
        self.clock_event = Clock.schedule_interval(self.update_frame, self.anim_delay)

    def dismiss(self):
        if self.clock_event:
            self.clock_event.cancel()
        
        self.popup.dismiss()
        

    def update_frame(self, dt):
        self.frame_index = (self.frame_index + 1) % len(self.frames)
        next_frame = self.frames[self.frame_index]
        self.image_widget.source = next_frame
        self.image_widget.reload()

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
    cap_counter = 0
    cap_percentage = 0.0
    new_playlist_name = ""
    is_android = BooleanProperty(platform == 'android')
    pitch_text = StringProperty()
    tempo_text = StringProperty()
    lod_text = StringProperty()
    processing = False
    CLAP_WAV_PATH = 'clap.wav'
    home_dir = os.getcwd()

    def __init__(self, **kw):
        # super().__init__(**kw)
        super(MainScreen, self).__init__()
        # self.ids.dropdown.values = os.listdir("songs")
        if platform == 'macosx':
            song_folders = os.listdir("songs")
            if '.DS_Store' in song_folders:
                song_folders.remove('.DS_Store')
                self.update_spinner_values(song_folders)
                print(f"Dropdown values: {self.ids.dropdown.values}")
        else:
            self.update_spinner_values(os.listdir("songs"))
            print(f"Dropdown values: {self.ids.dropdown.values}")
        print(platform)


        #### Testing
        # Disable video button and hide video if android device is used
        if platform == "win":
            self.toggle_video()
        self.ids.bg_video.state = "pause"
        self.ids.bg_video.opacity = 0
        self.remove_widget(self.ids.bg_video)
        self.ids.bg_video.texture = None

        self.pitch_text = f"Pitch (Semitione): 0"  # Initial value
        self.tempo_text = f"Tempo: {1.00}"  # Initial value
        self.lod_text = f"Difficulty: 0"  # Initial value

    def update_spinner_values(self, new_values):
        # Update the Spinner's values
        self.ids.dropdown.values = new_values

    def get_spinner_values(self):
        if platform == 'macosx':
            song_folders = os.listdir("songs")
            if '.DS_Store' in song_folders:
                song_folders.remove('.DS_Store')
            return song_folders + ["Add Playlist", "Remove Playlist"]
        else:
            return os.listdir("songs") + ["Add Playlist", "Remove Playlist"]

    def toggle_video(self):
        if platform != "android": 
            try:
                if self.ids.bg_video.state == "play":
                    self.ids.bg_video.state = "pause"
                    self.ids.bg_video.opacity = 0
                    self.remove_widget(self.ids.bg_video)
                    self.ids.bg_video.texture = None
                else:
                    self.ids.bg_video.state = "play"
                    self.ids.bg_video.opacity = 0.1
            except Exception as e:
                content = BoxLayout(orientation='vertical')
                popup = Popup(title='ERROR', 
                                content=content)
                scroll_error_1 = ScrollView(size_hint=(1, 0.8))
                scroll_error_1.add_widget(WrappedLabel(text = f"Unable to toggle video.\n{e}", size_hint=(1, None)))
                content.add_widget(scroll_error_1)
                error_btn_1 = Button(text='OK', on_press=popup.dismiss, size_hint_y=0.2)
                content.add_widget(error_btn_1)
                popup.open()
        
    def set_label_pitch(self, value):
        self.pitch_text = f"Pitch (Semitione): {round(value, 1)}"

    def set_label_tempo(self, value):
        self.tempo_text = f"Tempo: {round(value, 2)}"
        
    def set_label_lod(self, value):
        self.lod_text = f"Difficulty: {round(value, 1)}"

    def next_progress_step(self):
        # if self.ids.status_label.opacity == 0.0:
        #     self.ids.status_label.opacity = 1.0

        # if self.ids.progress.opacity == 0.0:
        #     self.ids.progress.opacity = 1.0

        # # self.ids.status_label.text = str(round((float(self.ids.status_label.text) + 1 / self.ids.progress.max), 2))
        # self.cap_percentage = (self.cap_counter + 1 / self.ids.progress.max) * 100
        # self.cap_counter += 1
        # self.ids.status_label.text = f"Loading Song: {round(self.cap_percentage, 2)}%"
        # if float(self.cap_percentage) >= 100:
        #     self.ids.status_label.text = "Done Loading Song"
        # self.ids.progress.value += 1
        pass

    def set_background_rbg(self):
        color_screen = self.manager.get_screen("color")
        try:
            with open('user_account.json', 'r') as f:
                # Loading the json data into a python object
                json_data = json.load(f)
            color_screen.ids.sliderR.value = json_data["background_color"][0] * 255
            color_screen.ids.sliderG.value = json_data["background_color"][1] * 255
            color_screen.ids.sliderB.value = json_data["background_color"][2] * 255
            color_screen.ids.sliderOpacity.value = json_data["background_color"][3]
        except Exception as e:
            print(f"Unable to pull up saved background color. Error: {e}")
            color_screen.ids.sliderR.value = 1 * 255
            color_screen.ids.sliderG.value = 0.83 * 255
            color_screen.ids.sliderB.value = 0 * 255
            color_screen.ids.sliderOpacity.value = 0.5

    def set_username(self):
        account_screen = self.manager.get_screen("account")
        try:
            with open('user_account.json', 'r') as f:
                # Loading the json data into a python object
                json_data = json.load(f)
            account_screen.ids.usernametext.text = json_data["username"]
        except Exception as e:
            print(f"Unable to pull up saved username. Error: {e}")

        # iOS doesn't support pandas due to C-extension compilation issues
        if platform == "ios":
            print("iOS detected: Skipping pandas-based statistics (C-extension unavailable)")
            # Set default/empty values for account statistics on iOS
            account_screen.ids.topcorrectsongs.text = "Top Song(s) Guessed Correctly: N/A"
            account_screen.ids.topincorrectsongs.text = "Top Song(s) Guessed Incorrectly: N/A"
            account_screen.ids.mostcommonkey.text = "Most Commonly Used Key Signature(s): N/A"
            account_screen.ids.mostcommontempo.text = "Most Commonly Used Tempo(s): N/A"
            account_screen.ids.mostcommonlod.text = "Most Commonly Used Level Of Difficulty(ies): N/A"
            account_screen.ids.averagescoreoverall.text = "Average Score (Overall): N/A"
            return

        tmp_dir = os.getcwd()
        try:
            try:
               os.chdir("data")
            except Exception as e:
                print(f"Data folder does not exist. Creating it now: {e}")
                os.mkdir("data") # Make directory if it does not exist
                os.chdir("data")
            if os.listdir() != []:
                data = []
                # df = pd.DataFrame(columns=["timestamp", "tempo", 
                #     "key", "level_of_difficulty", "random_song_chosen", 
                #     "user_guess", "user_score"])
                # Loop through all files in the directory
                for file_name in os.listdir():
                    if file_name.endswith(".json"):  # Check if the file is a JSON file
                        file_path = os.path.join(os.getcwd(), file_name)
                        with open(file_path, 'r') as file:
                            json_data = json.load(file)  # Load JSON data
                            data.append(json_data)      # Append to the list

                # Create a DataFrame from the list of JSON data
                df = pd.DataFrame(data)

                # Add columns to DataFrame
                df["correct_guess"] = [True if x >= 80 else False for x in df["user_score"]]

                # Step 1: Count song occurrences
                counts = df['random_song_chosen'].value_counts()

                # Step 2: Create a mapping of song → rank
                ranked = counts.rank(method='min', ascending=False).astype(int)
                total_songs = len(counts)

                # Step 3: Map each song to its popularity string
                popularity_map = {
                    song: f"{ranked[song]}/{total_songs}"
                    for song in counts.index
                }

                # Step 4: Create the new column
                df['song_popularity'] = df['random_song_chosen'].map(popularity_map)

                # Step 1: Count how often each song appears
                times_played_counts = df['random_song_chosen'].value_counts()

                # Step 2: Map those counts to a new column
                df['times_played'] = df['random_song_chosen'].map(times_played_counts)

                topcorrectsongs = df[df["correct_guess"] == True]["random_song_chosen"].mode().tolist()
                account_screen.ids.topcorrectsongs.text = "Top Song(s) Guessed Correctly:" + "\n".join(topcorrectsongs)
                print("; ".join(topcorrectsongs))

                topincorrectsongs = df[df["correct_guess"] == False]["random_song_chosen"].mode().tolist()
                account_screen.ids.topincorrectsongs.text = "Top Song(s) Guessed Incorrectly: " + "\n".join(topincorrectsongs)
                print("; ".join(topincorrectsongs))

                mostcommonkey = df["key"].mode().tolist()
                mostcommonkey = [str(val) for val in mostcommonkey]
                account_screen.ids.mostcommonkey.text = "Most Commonly Used Key Signature(s): " + "\n".join(mostcommonkey)
                print("; ".join(mostcommonkey))

                mostcommontempo = df["tempo"].mode().tolist()
                mostcommontempo = [str(val) for val in mostcommontempo]
                account_screen.ids.mostcommontempo.text = "Most Commonly Used Tempo(s): " + "\n".join(mostcommontempo)
                print("; ".join(mostcommontempo))

                mostcommonlod = df["level_of_difficulty"].mode().tolist()
                mostcommonlod = [str(val) for val in mostcommonlod]
                account_screen.ids.mostcommonlod.text = "Most Commonly Used Level Of Difficulty(ies): " + "\n".join(mostcommonlod)
                print("; ".join(mostcommonlod))

                averagescoreoverall = df["user_score"].sum() / len(df)
                account_screen.ids.averagescoreoverall.text = f"Average Score (Overall): {averagescoreoverall}"
                print(averagescoreoverall)

              

                # Display the DataFrame
                print(df)

                self.gridlayouttable = account_screen.ids.gridlayouttable
                # Using iterrows to iterate through rows
                for index, row in df.iterrows():
                    columns = ['timestamp', 'tempo', 'key', 'level_of_difficulty', 'random_song_chosen', 'user_guess', 'user_score',
                               'correct_guess', 'song_popularity', 'times_played']
                    for col in columns:
                        print(f"Index: {index}, {col}: {row[col]}")
                        self.gridlayouttable.add_widget(Label(text=str(row[col]), size_hint_y=None, height=20))
                

            os.chdir("..")
        except Exception as e:
            print(f"Error: {e}")
            os.chdir(tmp_dir)

    def create_new_playlist(self):
        # Remove periods, spaces, and special characters from playlist name
        relevant_chars = [',', ';', '.', ':', '(', ')', '\'', '\"', '[', ']', 
            ' ', '-', '=', '!', '@', '#', '$', '%', '^', '&', '*', '+']
        for char in relevant_chars:
            self.new_playlist_name.text = self.new_playlist_name.text.replace(char, '')

        if self.new_playlist_name.text != "":
            tmp_dir = os.getcwd()
            if platform == "ios":
                try:
                    from plyer import storagepath
                    documents_path = storagepath.get_documents_dir()
                    os.chdir("songs")
                    os.mkdir(self.new_playlist_name.text)
                    os.chdir("..")
                except Exception as e:
                    print(f"Error: {e}")
                    os.chdir(tmp_dir)
            else:
                try:
                    os.chdir("songs")
                    os.mkdir(self.new_playlist_name.text)
                    os.chdir("..")
                except Exception as e:
                    print(f"Error: {e}")
                    os.chdir(tmp_dir)

    def delete_playlist(self):
        # Remove periods, spaces, and special characters from playlist name
        relevant_chars = [',', ';', '.', ':', '(', ')', '\'', '\"', '[', ']', 
            ' ', '-', '=', '!', '@', '#', '$', '%', '^', '&', '*', '+']
        for char in relevant_chars:
            self.deleted_playlist_name.text = self.deleted_playlist_name.text.replace(char, '')

        if self.deleted_playlist_name.text != "":
            tmp_dir = os.getcwd()
            try:
                os.chdir("songs")
                # os.rmdir(self.deleted_playlist_name.text)
                shutil.rmtree(self.deleted_playlist_name.text)
                os.chdir("..")
            except Exception as e:
                print(f"Error: {e}")
                os.chdir(tmp_dir)
                content = BoxLayout(orientation='vertical')
                popup = Popup(title='ERROR', 
                            content=content)
                scroll_error_1 = ScrollView(size_hint=(1, 0.8))
                scroll_error_1.add_widget(WrappedLabel(text = f"Unable to remove playlist.\nPlease ensure that playlist exists before deleting.", size_hint=(1, None)))
                content.add_widget(scroll_error_1)
                error_btn_1 = Button(text='OK', on_press=popup.dismiss, size_hint_y=0.2)
                content.add_widget(error_btn_1)
                popup.open()

        else:
            content = BoxLayout(orientation='vertical')
            popup = Popup(title='ERROR', 
                        content=content)
            scroll_error_1 = ScrollView(size_hint=(1, 0.8))
            scroll_error_1.add_widget(WrappedLabel(text = f"Unable to remove playlist.\nPlease ensure that playlist exists before deleting.", size_hint=(1, None)))
            content.add_widget(scroll_error_1)
            error_btn_1 = Button(text='OK', on_press=popup.dismiss, size_hint_y=0.2)
            content.add_widget(error_btn_1)
            popup.open()

    def toggle_view_playlist(self):
        self.ids.view_playlist.disabled = False
        self.ids.playbtn.disabled = False
        # self.ids.pausebtn.disabled = False
        # self.ids.restartbtn.disabled = False
        self.song_selected = False

        if self.ids.dropdown.text == "Special":
            self.ids.sliderA.value = 0
            self.ids.sliderA.disabled = True
            self.ids.slider1.value = 1
            self.ids.slider1.disabled = True
            self.ids.slider2.value = 0
            self.ids.slider2.disabled = True
            tmp_song_dir = os.path.join(home_app_dir, 'songs', self.ids.dropdown.text)
            list_songs = [song for song in os.listdir(tmp_song_dir) if song != 'temp.wav' or song != 'temp_original.wav']
            if list_songs == []:
                self.ids.playbtn.disabled = True

        elif self.ids.dropdown.text == "Add Playlist":
            content = BoxLayout(orientation='vertical')
            popup = Popup(title='Enter Playlist Name', 
                            content=content)
            scroll_error_1 = ScrollView(size_hint=(1, 0.8))
            scroll_error_1.add_widget(WrappedLabel(text = f"Please enter a new playlist name below", size_hint=(1, None)))
            content.add_widget(scroll_error_1)
            self.new_playlist_name = TextInput(hint_text="Playlist name goes here")
            content.add_widget(self.new_playlist_name)
            error_btn_1 = Button(text='OK', on_press=lambda x: (self.create_new_playlist(), self.update_spinner_values(os.listdir("songs") + ["Add Playlist", "Remove Playlist"]), popup.dismiss()), size_hint_y=0.2)
            self.ids.dropdown.text = "Select Playlist"
            self.ids.playbtn.disabled = True
            content.add_widget(error_btn_1)
            popup.open()
        elif self.ids.dropdown.text == "Remove Playlist":  
            content = BoxLayout(orientation='vertical')
            popup = Popup(title='Delete Playlist', 
                            content=content)
            scroll_error_1 = ScrollView(size_hint=(1, 0.8))
            scroll_error_1.add_widget(WrappedLabel(text = f"Please enter the name of the playlist that will be deleted", size_hint=(1, None)))
            content.add_widget(scroll_error_1)
            self.deleted_playlist_name = TextInput(hint_text="Playlist name that will be deleted goes here")
            content.add_widget(self.deleted_playlist_name)
            error_btn_1 = Button(text='OK', on_press=lambda x: (self.delete_playlist(), self.update_spinner_values(os.listdir("songs") + ["Add Playlist", "Remove Playlist"]), popup.dismiss()), size_hint_y=0.2)
            self.ids.dropdown.text = "Select Playlist"
            self.ids.playbtn.disabled = True
            content.add_widget(error_btn_1)
            popup.open()
        else:
            if self.ids.sliderA.disabled:
                self.ids.sliderA.disabled = False
            if self.ids.slider1.disabled:
                self.ids.slider1.disabled = False
            if self.ids.slider2.disabled:
                self.ids.slider2.disabled = False
            tmp_song_dir = os.path.join(home_app_dir, 'songs', self.ids.dropdown.text)
            list_songs = [song for song in os.listdir(tmp_song_dir) if song != 'temp.wav' or song != 'temp_original.wav']
            if list_songs == []:
                self.ids.playbtn.disabled = True

        
    def play_random_song(self):

        print(f'Song selected: {self.song_selected}')

        # Disable sliders and dropdown
        self.ids.sliderA.disabled = True
        self.ids.slider1.disabled = True
        self.ids.slider2.disabled = True

        if self.song_selected == False:
            try:
                self.loading_popup = None
            except Exception as e:
                print(f"Could not unload self.loading_popup. Error: {e}")

            
            self.loading_popup = AnimatedPopup()
            self.loading_popup.open()


            # anim_img = Image(source='Loading.gif', anim_delay=0.05)
            # self.loading_popup = Popup(title='Loading Song', content=anim_img, size_hint=(0.6, 0.6))
            # self.loading_popup.open()

            
            # Reset sound variable
            try:
                self.sound.unload()
            except Exception as e:
                print(f"Could not unload self.sound. Error: {e}")
            self.sound = None

            self.ids.playbtn.disabled = True
            self.ids.submitbtn.disabled = False
            self.ids.dropdown.disabled = True
            self.ids.view_playlist.disabled = True
            # self.ids.pausebtn.disabled = False
            self.ids.restartbtn.disabled = True
            dropdown_text = self.ids.dropdown.text
            dropdown_text = dropdown_text.replace(' ', '')

            

            print(f'Before chdir into songs')
            print(os.listdir())
            os.chdir('songs')
            tmp_directory = os.getcwd()
            print(f'After chdir into songs')
            print(os.listdir())
            # os.chdir(dropdown_text)
            print(dropdown_text)
            # if 'temp.wav' in os.listdir():
            #     os.remove('temp.wav')
            for _ in range(5):
                try:
                    print("*****************FOR LOOP************")
                    print(os.listdir())
                    # music_dirs = ["Hymns", "NurseryRhymes", "Custom", "Special", "Add Playlist"]
                    music_dirs = os.listdir()
                    # Add "Add Playlist" & "Remove Playlist" options to "Playlist Options"
                    for directory in music_dirs:
                        try:
                            # Skip over '.DS_Store' directory
                            if directory == '.DS_Store':
                                continue

                            os.chdir(directory)
                        except Exception as e:
                            print(f"Directory does not exist. Creating it now: {e}")
                            os.mkdir(directory) # Make directory if it does not exist
                        if 'temp.mp3' in os.listdir():
                            os.remove('temp.mp3')
                        elif 'temp.m4a' in os.listdir():
                            os.remove('temp.m4a')
                        elif 'temp.wav' in os.listdir():
                            os.remove('temp.wav')
                        os.chdir("..")
                    break
                except PermissionError as e:
                    print(f"Permission Error (will try again): {e}")
                    os.chdir(tmp_directory)
                    time.sleep(0.5)
            
            self.next_progress_step()
            os.chdir(dropdown_text)
            list_songs = [song for song in os.listdir() if song != 'temp.wav' or song != 'temp_original.wav']

            if list_songs == []:
                content = BoxLayout(orientation='vertical')
                popup = Popup(title='ERROR', 
                            content=content)
                scroll_error_1 = ScrollView(size_hint=(1, 0.8))
                scroll_error_1.add_widget(WrappedLabel(text = f"No song available.\nTry to add a song first.", size_hint=(1, None)))
                content.add_widget(scroll_error_1)
                error_btn_1 = Button(text='OK', on_press=popup.dismiss, size_hint_y=0.2)
                content.add_widget(error_btn_1)
                popup.open()
                self.cap_percentage = 100

                # Reset after exception occurs
                self.ids.playbtn.disabled = False
                self.ids.submitbtn.disabled = True
                self.ids.sliderA.disabled = False
                self.ids.slider1.disabled = False
                self.ids.slider2.disabled = False
                self.ids.sliderA.disabled = False
                self.ids.dropdown.disabled = False
                self.ids.view_playlist.disabled = False

                # self.ids.status_label.text = "0"
                # self.ids.progress.value = 0
                # self.ids.status_label.opacity = 0.0
                # self.ids.progress.opacity = 0.0
                self.ids.dropdown.disabled = False
                self.ids.view_playlist.disabled = False
                os.chdir('..')
                os.chdir('..')
                return

            random_song = randint(0, len(list_songs) - 1)
            print(f'Song: {list_songs[random_song]}')
            print(f'Tempo: {self.ids.slider1.value}')
            print(f'Pitch: {self.ids.sliderA.value}')
            print(f'Level of difficulty: {self.ids.slider2.value}')

            try:
                if platform == "android":
                    # Clock.schedule_once(lambda dt: self.change_audio_properties(list_songs[random_song], f'{list_songs[random_song]}_{self.ids.slider1.value}_{self.ids.sliderA.value}_{self.ids.slider2.value}.mp3', self.ids.slider1.value, self.ids.sliderA.value, self.ids.slider2.value, True), 0.1)
                    Clock.schedule_once(lambda dt: self.change_audio_properties(list_songs[random_song], f'{list_songs[random_song]}', self.ids.slider1.value, self.ids.sliderA.value, self.ids.slider2.value), 0.1)
                else:
                    import threading
                    if not self.processing:
                        self.processing = True
                        threading.Thread(target=self.change_audio_properties, args=[list_songs[random_song], f'{list_songs[random_song]}', self.ids.slider1.value, self.ids.sliderA.value, self.ids.slider2.value]).start()
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
                self.ids.sliderA.disabled = False
                self.ids.dropdown.disabled = False
                self.ids.view_playlist.disabled = False

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
    
    def detect_pitch_autocorr(self, y, sr, fmin=80, fmax=1500):
        """Return fundamental freq via autocorrelation in y."""
        # full autocorr, then keep positive lags
        corr = np.correlate(y, y, mode='full')
        corr = corr[corr.size//2:]
        # lag bounds
        min_lag = int(sr / fmax)
        max_lag = int(sr / fmin)
        if max_lag >= len(corr):
            max_lag = len(corr)-1
        segment = corr[min_lag:max_lag]
        if segment.size == 0:
            return 0
        lag = segment.argmax() + min_lag
        return sr/lag if lag>0 else 0
    
    def safe_read(self, reader, buffer):
        temp = np.zeros(buffer.shape, dtype=buffer.dtype)
        n = reader.read(temp)
        if n < buffer.shape[1]:
            # Already padded with zeros
            return n
        elif n > buffer.shape[1]:
            # Truncate
            temp = temp[:, :buffer.shape[1]]
            np.copyto(buffer, temp)
            return buffer.shape[1]
        else:
            np.copyto(buffer, temp)
            return n

    def change_audio_properties(
        self, audio_path, output_path,
        tempo_factor=1.0, pitch_semitones=0, remove_pitch_count=0
    ):
        tmp_dir = os.getcwd()
        tmp2 = "temp.wav"
        
        # iOS handling: skip complex audio processing due to subprocess limitations
        if platform == "ios":
            print("iOS detected: Using simplified audio playback (no processing)")
            self.loading_popup.dismiss()
            # Directly load and play the original audio file using SoundLoader
            try:
                audio_file_path = os.path.join(os.getcwd(), audio_path)
                self.sound = SoundLoader.load(audio_file_path)
                if self.sound:
                    self.sound.play()
                    self.sound.bind(on_stop=self.on_song_end)
                    self.song_selected = True
                    self.ids.playbtn.disabled = True
                    self.ids.pausebtn.disabled = False
                    self.ids.restartbtn.disabled = False
                    self.ids.submitbtn.disabled = False
                else:
                    raise Exception("Failed to load audio file")
            except Exception as e:
                print(f"Error loading audio on iOS: {e}")
                content = BoxLayout(orientation='vertical')
                popup = Popup(title='ERROR', content=content)
                scroll_error = ScrollView(size_hint=(1, 0.8))
                scroll_error.add_widget(WrappedLabel(text=f"Unable to play song on iOS.\nError: {e}", size_hint=(1, None)))
                content.add_widget(scroll_error)
                error_btn = Button(text='OK', on_press=popup.dismiss, size_hint_y=0.2)
                content.add_widget(error_btn)
                popup.open()
                self.ids.playbtn.disabled = False
                self.ids.submitbtn.disabled = True
                self.ids.sliderA.disabled = False
                self.ids.slider1.disabled = False
                self.ids.slider2.disabled = False
                self.ids.dropdown.disabled = False
                self.ids.view_playlist.disabled = False
            finally:
                os.chdir('..')
                os.chdir('..')
            return
        
        y, sr = sf.read(audio_path)

        # Pitch classes to remove
        removal_order = [8, 10, 3, 6, 1, 4, 7, 9, 11, 0, 2, 5]
        print(remove_pitch_count)
        removal = set(removal_order[:int(remove_pitch_count)])

        # pick which pitch‐classes to remove: 0=C, 1=C#, … 5=F, 7=G, … 11=B
        # e.g. remove F (5) and G (7) below
        # remove_notes = [5, 7]

        # === Step 1: Load audio & raw samples ===
        audio    = AudioSegment.from_file(audio_path, format='mp3')
        sr       = audio.frame_rate
        channels = audio.channels
        sw       = audio.sample_width

        # to mono float32 in [-1,1]
        raw = np.array(audio.get_array_of_samples())
        if channels > 1:
            raw = raw.reshape(-1, channels).mean(axis=1)
        max_val = float(1 << (8*sw - 1))
        samples = raw.astype(np.float32) / max_val

        # === Step 2: Onset detection via energy diff ===
        frame_size = 2048
        hop_size   = 512
        n_frames   = 1 + (len(samples) - frame_size) // hop_size

        # compute energies
        energies = np.empty(n_frames, dtype=np.float32)
        for i in range(n_frames):
            start = i * hop_size
            frame = samples[start:start+frame_size]
            energies[i] = np.sum(frame*frame)

        # positive diffs and threshold
        diff      = np.diff(energies)
        diff[diff < 0] = 0
        threshold = diff.mean() * 4
        onsets   = np.where(diff > threshold)[0] + 1

        # convert to sample indices and build segments
        onset_samples = (onsets * hop_size).astype(int)
        segments = []
        for i, s in enumerate(onset_samples):
            e = onset_samples[i+1] if i+1 < len(onset_samples) else len(samples)
            segments.append((s, e))

        try:
            os.chdir("..")
            os.chdir("..")
            clap   = AudioSegment.from_file(self.CLAP_WAV_PATH, format='wav')
            out  = audio[:]  # copy
            os.chdir(tmp_dir)
        except Exception as e:
            print(f"Error: {e}")
            os.chdir(tmp_dir)

        for start_s, end_s in segments:
            frame = samples[start_s:end_s]
            # skip too-short
            if len(frame) < frame_size//2:
                continue
            freq = self.detect_pitch_autocorr(frame, sr)
            if freq <= 0:
                continue

            # midi number and pitch‐class
            midi = 69 + 12 * math.log2(freq/440.0)
            pc   = int(round(midi)) % 12

            if pc in removal:
                # compute ms positions
                start_ms = int(1000 * start_s / sr)
                end_ms   = int(1000 * end_s   / sr)
                # mute original segment
                sil = AudioSegment.silent(duration=(end_ms - start_ms))
                out = out[:start_ms] + sil + out[end_ms:]
                # overlay clap at segment start
                out = out.overlay(clap, position=start_ms)
        
        # === Step 5: Export ===
        # Convert to mono
        out_final = out.set_channels(1)
        out_final.export("temp.wav", format='wav')
        AudioSegment.from_wav(tmp2).export(tmp2, format="wav", parameters=["-acodec", "pcm_s16le"])

        # Change tempo of song if necessary
        if tempo_factor != 1.0:
            shutil.copy('temp.wav', 'temp_original.wav')
            
            input_path = 'temp_original.wav'
            output_path = "temp.wav"

            # Attempt 1: Set frame_length to 2048, as it's the "larger" shape in the error
            # This assumes the reader is providing 2048 samples, and audiotsm needs to match that.
            try:
                print(f"Attempting with frame_length = 2048...")
                with WavReader(input_path) as reader:
                    with WavWriter(output_path, reader.channels, reader.samplerate) as writer:
                        tsm = phasevocoder(reader.channels, speed=tempo_factor, frame_length=2048)
                        tsm.run(reader, writer)
                print("Processing successful with frame_length = 2048")

            except ValueError as e:
                print(f"Error with frame_length = 2048: {e}")
                # If the above fails, try the other value from the error message.
                # Attempt 2: Set frame_length to 1024
                print(f"Attempting with frame_length = 1024...")
                try:
                    with WavReader(input_path) as reader:
                        with WavWriter(output_path, reader.channels, reader.samplerate) as writer:
                            tsm = phasevocoder(reader.channels, speed=tempo_factor, frame_length=1024)
                            tsm.run(reader, writer)
                    print("Processing successful with frame_length = 1024")
                except ValueError as e_inner:
                    print(f"Error with frame_length = 1024: {e_inner}")
                    print("Neither 1024 nor 2048 worked directly as frame_length.")
                    print("The issue might be more nuanced or related to the WavReader's internal buffering.")

            # Check if the file exists before attempting to remove it
            if os.path.exists(input_path):
                os.remove(input_path)

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

        # Part 5
        self.next_progress_step()

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

        # Part 6
        self.next_progress_step()

        # self.sound = SoundLoader.load(output_path)
        print(f"tmp2: {tmp2}")
        print(f"os.getcwd() -> {os.getcwd()}")

        
        self.loading_popup.dismiss()
        if platform == "android":
            print("Before calling: play_aac_file")
            self.play_aac_file(f"{os.getcwd()}/{tmp2}")
            print("After calling: play_aac_file")
        elif platform == "ios":
            print("Before calling: self.sound = SoundLoader.load(tmp2) [iOS]")
            self.sound = SoundLoader.load(f"{os.getcwd()}/{tmp2}")
            print("After calling: self.sound = SoundLoader.load(tmp2) [iOS]")
            self.sound.play()
            self.sound.bind(on_stop=self.on_song_end)
        elif platform == "macosx":
            print("Before calling: self.sound = SoundLoader.load(tmp2)")
            self.sound = SoundLoader.load(f"{os.getcwd()}/{tmp2}")
            print("After calling: self.sound = SoundLoader.load(tmp2)")
            self.sound.play()
            self.sound.bind(on_stop=self.on_song_end)
        else:      
            print("Before calling: self.sound = SoundLoader.load(tmp2)")
            self.sound = SoundLoader.load(f"{os.getcwd()}\\{tmp2}")
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
        try:
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
        except Exception as e:
            print(f"Sorry, unable to add song. Error: {e}")

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
    is_android = BooleanProperty(platform == 'android')
    def getscore(self):
        main_screen = self.manager.get_screen("main")
        guess_input_text = main_screen.ids.guess_input.text
        title_text = main_screen.random_song
        main_screen.song_selected = False

        try:
            main_screen.pause_song()
        except Exception as e:
            print("There was an error when attempting to stop the song\n{e}")

        title_text = title_text.replace('_', ' ')

        # Remove empty spaces at beginning or end of the strings
        guess_input_text_stripped = guess_input_text.strip()
        title_text_stripped = title_text.strip()

        # Remove special characters from users guess
        relevant_chars = [',', ';', '.', ':', '(', ')', '\'', '\"', '[', ']']
        for char in relevant_chars:
            guess_input_text_stripped = guess_input_text_stripped.replace(char, '')


        #Show percentage of difference between two strings
        similarity_score = SequenceMatcher(None, guess_input_text_stripped.lower(), title_text_stripped.lower()).ratio() * 100

        # score_screen = self.manager.get_screen("score")
        self.ids.scorelabel.text = f"Song Title: {title_text}\nUser's Guess: {guess_input_text}\nSimilarity Score: {similarity_score}"

        user_log = {'timestamp':datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S"), 'tempo': main_screen.ids.slider1.value, 'key': main_screen.ids.sliderA.value, 
      'level_of_difficulty': main_screen.ids.slider2.value, 
      'random_song_chosen': title_text, 
      'user_guess':guess_input_text,
      'user_score': similarity_score}
        print("User log: ")
        print(user_log)

        if platform == "android":
            tmp_directory = os.getcwd()
            try:
                try:
                    os.chdir("data")
                except Exception as e:
                    print(f"Data folder does not exist. Creating it now: {e}")
                    os.mkdir("data") # Make directory if it does not exist
                    os.chdir("data")
                # Write dictionary to a JSON file
                with open(f"{user_log['timestamp']}_{uuid.uuid4()}.json", "w") as json_file:
                    json.dump(user_log, json_file, indent=4)  # 'indent' makes the JSON file
                os.chdir("..")
            except Exception as e:
                print(f"Error: {e}")
                os.chdir(tmp_directory)
        elif platform == "ios":
            try:
                from plyer import storagepath
                documents_path = storagepath.get_documents_dir()
                data_path = os.path.join(documents_path, "data")
                if not os.path.exists(data_path):
                    os.makedirs(data_path)
                # Write dictionary to a JSON file
                with open(os.path.join(data_path, f"{user_log['timestamp']}_{uuid.uuid4()}.json"), "w") as json_file:
                    json.dump(user_log, json_file, indent=4)  # 'indent' makes the JSON file 
            except Exception as e:
                print(f"Error writing user log on iOS: {e}")
        else:
            # Write dictionary to a JSON file
            with open(f".\\data\\{user_log['timestamp']}_{uuid.uuid4()}.json", "w") as json_file:
                json.dump(user_log, json_file, indent=4)  # 'indent' makes the JSON file 

        main_screen.ids.guess_input.text = ""
        main_screen.ids.playbtn.text = "Play"
        main_screen.ids.submitbtn.disabled = True
        main_screen.ids.sliderA.disabled = False
        main_screen.ids.slider1.disabled = False
        main_screen.ids.slider2.disabled = False
        # main_screen.ids.status_label.text = "0"
        # main_screen.ids.progress.value = 0
        # main_screen.ids.status_label.opacity = 0.0
        # main_screen.ids.progress.opacity = 0.0
        main_screen.ids.dropdown.disabled = False
        main_screen.ids.view_playlist.disabled = False

    def restart_app(self):
        import sys
        # Restart the script
        python = sys.executable
        os.execl(python, python, *sys.argv)

class AccountScreen(Screen):
    def pass_this(self):
        pass

class PlaylistDeleteScreen(Screen):
    def pass_this(self):
        pass

class ColorScreen(Screen):
    def change_theme(self):
        # color_screen = self.manager.get_screen("color")
        # self.background_color = [color_screen.ids.sliderR.value / 255, color_screen.ids.sliderG.value / 255, color_screen.ids.sliderB.value / 255, color_screen.ids.sliderOpacity.value]

        # gts = GuessThatSongApp()
        # # gts.change_theme(color_screen.ids.sliderR.value / 255, color_screen.ids.sliderG.value / 255, color_screen.ids.sliderB.value / 255, color_screen.ids.sliderOpacity.value)
        # Clock.schedule_once(lambda dt: gts.change_theme(color_screen.ids.sliderR.value / 255, color_screen.ids.sliderG.value / 255, color_screen.ids.sliderB.value / 255, color_screen.ids.sliderOpacity.value), 0.1)
        # # print(color_screen.ids.sliderR.value)
        pass

class MyScreenManager(ScreenManager):
    pass

class GuessThatSongApp(App):
    # Properties to hold the current theme and background color.
    theme = StringProperty('light')  # 'light' or 'dark'
    # background_color = ListProperty([0.678, 0.847, 0.902, 1])  # light blue RGBA
    try:
        with open('user_account.json', 'r') as f:
            # Loading the json data into a python object
            json_data = json.load(f)
            background_color = ListProperty(json_data["background_color"])
    except Exception as e:
        print(f"Could not read successfully user_account.json file. Error: {e}")
        background_color = ListProperty([1, 0.83, 0, 0.5])  # magenta RGBA

    def toggle_theme(self):
        # Toggle between light mode (light blue) and dark mode (dark blue)
        if self.theme == 'light':
            self.theme = 'dark'
            # self.background_color = [0, 0, 0.5, 1]  # dark blue RGBA
            self.background_color = [0.74, 0.09, 0.25, 1] # magenta RGBA
        else:
            self.theme = 'light'
            # self.background_color = [0.678, 0.847, 0.902, 1]
            self.background_color = [1, 0.83, 0, 0.5] # yellow RGBA

    def change_theme(self, r, g, b, opacity):
        self.background_color = [r/255, g/255, b/255, opacity]
        background_color = self.background_color
        if platform == "ios":
            try:
                from plyer import storagepath
                documents_path = storagepath.get_documents_dir()
                if "user_account.json" not in os.listdir():
                    json_data = {"background_color": self.background_color}
                    # Write dictionary to a JSON file
                    with open(os.path.join(documents_path, 'user_account.json'), "w") as json_file:
                        json.dump(json_data, f)
                else:
                    with open(os.path.join(documents_path, 'user_account.json'), 'r') as f:
                        # Loading the json data into a python object
                        json_data = json.load(f)
                    json_data["background_color"] = self.background_color
                    with open(os.path.join(documents_path, 'user_account.json'), 'w') as f:
                        json.dump(json_data, f)
            except Exception as e:
                print(f"Error writing user account file on iOS: {e}")
        else:
            if "user_account.json" not in os.listdir():
                json_data = {"background_color": self.background_color}
                with open('user_account.json', 'w') as f:
                    json.dump(json_data, f)
            else:
                with open('user_account.json', 'r') as f:
                    # Loading the json data into a python object
                    json_data = json.load(f)
                json_data["background_color"] = self.background_color
                with open('user_account.json', 'w') as f:
                    json.dump(json_data, f)

    def change_username(self, username):
        if platform == "ios":
            try:
                from plyer import storagepath
                documents_path = storagepath.get_documents_dir()
                if "user_account.json" not in os.listdir():
                    json_data = {"username": username}
                    # Write dictionary to a JSON file
                    # with open(os.path.join(documents_path, 'user_account.json'), "w") as json_file:
                    #     json.dump(json_data, f)
                    with open('user_account.json', 'w') as f:
                        json.dump(json_data, f)
                else:
                    # with open(os.path.join(documents_path, 'user_account.json'), 'r') as f:
                    with open('user_account.json', 'w') as f:
                        # Loading the json data into a python object
                        json_data = json.load(f)
                    json_data["username"] = username
                    # with open(os.path.join(documents_path, 'user_account.json'), 'w') as f:
                    with open('user_account.json', 'w') as f:
                        json.dump(json_data, f)
            except Exception as e:
                print(f"Error writing user account file on iOS: {e}")
        else:
            if "user_account.json" not in os.listdir():
                json_data = {"username": username}
                with open('user_account.json', 'w') as f:
                    json.dump(json_data, f)
            else:
                with open('user_account.json', 'r') as f:
                    # Loading the json data into a python object
                    json_data = json.load(f)
                json_data["username"] = username
                with open('user_account.json', 'w') as f:
                    json.dump(json_data, f)
                    
    def build(self):
        print(platform)
        return MyScreenManager()
    
    def on_start(self):
        pass

if __name__ == '__main__':
    GuessThatSongApp().run()