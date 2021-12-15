# Extract audio

# https://github.com/AudioCommons/timbral_models
import timbral_models

import sys
from re import split
from math import sqrt
import moviepy.editor as mp
from numpy.lib.function_base import trim_zeros
import librosa
import librosa.display
import numpy as np
from matplotlib import cm
from matplotlib.legend_handler import HandlerLine2D
import matplotlib.pyplot as plt
from os import walk
from typing import List, Dict

# Draw QQ-Plot
import scipy.stats as stats

N_FFT = 256*2*2*2*2
WIN_LENGTH = 256*2*2*2*2
HOP_LENGTH = 128*2*2*2

GOOD = 200
BAD = 404
SERVE_LENGTH_MIN, SERVE_LENGTH_MAX = 18.286, sqrt(18.286*18.286 + 4.1148*4.1148) # unit is m (=meter)
NET_SERVE_LENGTH_MIN, NET_SERVE_LENGTH_MAX = 11.886, sqrt(11.886*11.886 + 4.1148*4.1148) # unit is m (=meter)

counter = 100

def colourSpliter(size):
    jump = int((0xFF - 0x10) / size)
    return ["#"+hex(x)[2:]+hex(x)[2:]+hex(x)[2:] for x in range(0x10, 0xFF, jump)]

def frameToSpeed(frame):
    flight_time_of_ball = float(frame) / 30.0
    speed_min = SERVE_LENGTH_MIN * 3600.0 / flight_time_of_ball / 1000.0
    speed_max = SERVE_LENGTH_MAX * 3600.0 / flight_time_of_ball / 1000.0
    return speed_min, speed_max

def net_frameToSpeed(frame):
    flight_time_of_ball = float(frame) / 30.0
    speed_min = NET_SERVE_LENGTH_MIN * 3600.0 / flight_time_of_ball / 1000.0
    speed_max = NET_SERVE_LENGTH_MAX * 3600.0 / flight_time_of_ball / 1000.0
    return speed_min, speed_max

class RawData:
    path: str
    
    goodOrBad: int
    serveType: str
    frame: int
    speed_min: float
    speed_max: float

    net: bool

    audio_array: np.ndarray
    max_index, min_index = None, None

    # Timbre
    timbre: List[Dict[str, float]]

    def __init__(self, dirname: str, filename: str) -> None:
        self.path = dirname+filename
        self.frame = 1

        # Parse infomation
        filename = filename.replace(" - ", ",")
        filename = filename.replace(".mp4", "")
        if "(" in filename:
            filename = filename.replace(" (", "-")
            filename = filename.replace(")", "")
            temp = filename.split("-")
            self.frame = temp[1]
            filename = temp[0]
        self.serveType = filename.split(",")[1]

        filename = filename.replace(" ", "")
        filename = filename.replace("-", ",")
        filename = filename.replace("(", ",")
        filename = filename.replace(").", ",")
        splited = split(",", filename)
        
        if "good" not in splited[0]:
            self.goodOrBad = BAD
        else:
            self.goodOrBad = GOOD

        if "net" not in splited[1]:
            self.net = False
        else:
            self.net = True
            
        self.serveType = splited[1]
        
        if self.net == True:
            self.speed_min, self.speed_max = net_frameToSpeed(self.frame)
        else:
            self.speed_min, self.speed_max = frameToSpeed(self.frame)

        #self.frame = int(splited[2])
        #self.speed_min, self.speed_max = frameToSpeed(splited[2])
        # print((speed_max + speed_min) / 2.0)

        clip = mp.VideoFileClip(self.path)
        self.audio_array = clip.audio.to_soundarray()
        self.audio_array = (self.audio_array[:,0] + self.audio_array[:,1]) / 2
        self.audio_array = np.array(self.audio_array)
        # minimum = abs(self.audio_array.min())
        self.max_index, self.min_index = self.audio_array.argmax(), self.audio_array.argmin()
        self.audio_array = (self.audio_array[self.max_index-1000:self.max_index+1500])

        # Extract Timbre
        self.timbre = timbral_models.timbral_extractor(self.audio_array, fs=44100, verbose=False)

    def getAverageSpeed(self) -> float:
        return ((self.speed_max + self.speed_min) / 2.0)

    def getTimbreAsFloatArray(self) -> list():
        return [
            self.timbre["hardness"],
            self.timbre["depth"],
            self.timbre["brightness"],
            self.timbre["roughness"],
            self.timbre["warmth"],
            self.timbre["sharpness"],
            self.timbre["boominess"],
            self.timbre["reverb"],
            ]

    def myTimbreSelection(self) -> float:
        return (self.timbre["warmth"] + self.timbre["boominess"])

    def printInfo(self):
        print("Path", self.path)
        print("Speed", (self.speed_max + self.speed_min) / 2.0)

    def drawSoundPlot(self):
        plt.title(self.path.split('/')[-1].split('.mp4')[0])
        plt.plot(self.audio_array)

    def drawMelSpectrogram(self):
        stft_result = librosa.stft(self.audio_array, n_fft=N_FFT, win_length = WIN_LENGTH, hop_length=HOP_LENGTH)
        D = np.abs(stft_result)
        S_dB = librosa.power_to_db(D, ref=np.max)
        librosa.display.specshow(S_dB, sr=44100, hop_length = HOP_LENGTH, y_axis='linear', x_axis='time', cmap = cm.jet)
        plt.title(self.path.split('/')[-1].split('.mp4')[0])
        # plt.colorbar(format='%2.0f dB')
        plt.savefig("./MelSpectrogram/"+ self.path.split('/')[-1].split('.mp4')[0] + ".png")
        plt.cla()

    def drawQQPlot(self):
        stats.probplot(self.audio_array, dist=stats.norm, plot=plt)
        plt.show()

good: List[RawData] = []
bad: List[RawData] = []

for (dirpath, dirnames, filenames) in walk("./video_data/serve/bad/"):
    for filename in filenames:
        nowData = RawData(dirpath, filename)
        bad.append(nowData)
    break

for (dirpath, dirnames, filenames) in walk("./video_data/serve/good/"):
    for filename in filenames:
        nowData = RawData(dirpath, filename)
        good.append(nowData)
    break

good_counter = []
bad_counter = []
STANDARD = -18.0


for each in good:
    stft_result = librosa.stft(each.audio_array, n_fft=N_FFT, win_length = WIN_LENGTH, hop_length=HOP_LENGTH)
    D = np.abs(stft_result)
    S_dB = librosa.power_to_db(D, ref=np.max)
    flatten_arr = np.array(S_dB).flatten()
    counter = 0
    for x in flatten_arr:
        if(x >= STANDARD):
            counter = counter + 1
    good_counter.append(counter)
    #librosa.display.specshow(S_dB, sr=44100, hop_length = 1024, y_axis='linear', x_axis='time', cmap = cm.jet)
    #plt.title(each.path.split('/')[-1].split('.mp4')[0])
    #plt.colorbar(format='%2.0f dB')
    #plt.show()

for each in bad:
    stft_result = librosa.stft(each.audio_array, n_fft=N_FFT, win_length = WIN_LENGTH, hop_length=HOP_LENGTH)
    D = np.abs(stft_result)
    S_dB = librosa.power_to_db(D, ref=np.max)
    flatten_arr = np.array(S_dB).flatten()
    counter = 0
    for x in flatten_arr:
        if(x >= STANDARD):
            counter = counter + 1
    bad_counter.append(counter)
    #librosa.display.specshow(S_dB, sr=44100, hop_length = 1024, y_axis='linear', x_axis='time', cmap = cm.jet)
    #plt.title(each.path.split('/')[-1].split('.mp4')[0])
    #plt.colorbar(format='%2.0f dB')
    #plt.show()


for x in good:
    #x.drawMelSpectrogram()
    print(x.frame, ",", x.getAverageSpeed(), ",",x.getTimbreAsFloatArray())
for x in bad:
    #x.drawMelSpectrogram()
    print(x.frame, ",", x.getAverageSpeed(), ",",x.getTimbreAsFloatArray())


"""
good_avg = 0.0
bad_avg = 0.0
for x in good_counter:
    good_avg = good_avg + x
for x in bad_counter:
    bad_avg = bad_avg + x

print(good_avg / len(good_counter), good_counter)
print(bad_avg / len(bad_counter), bad_counter)
"""


"""
colours=["red", "green", "blue", "orange", "crimson", "yellow"]
f1 = plt.figure("good")
plt.title("good")
for i in range(len(good)):
    plt.plot(good[i].audio_array, label=good[i].path.split('/')[-1].split('.mp4')[0], c="red", alpha=0.4)
f2 = plt.figure("bad")
plt.title("bad")
for x in bad:
    plt.plot(x.audio_array, label=x.path.split('/')[-1].split('.mp4')[0], c="blue", alpha=0.4)
plt.show()
"""