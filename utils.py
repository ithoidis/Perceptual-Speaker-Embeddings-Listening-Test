"""
@Author: Iordanis Thoidis
@Date: 11/5/23
@Link: https://github.com/ithoidis/Perceptual-Speaker-Embeddings-Listening-Test
"""

import numpy as np
import psychopy
import librosa
from scipy.io.wavfile import read, write
from pydub import AudioSegment
# import soundfile as sf


def select_audio_device(device):
    assert device in ['PTB', 'sounddevice', 'pyo', 'pygame']
    psychopy.prefs.hardware['audioLib'] = [device]
    psychopy.prefs.saveUserPrefs()


def attenuate_sound(x, level=0.):
    return np.array(x) * 10.**(level/20)


def silence(duration, fs):
    return np.zeros(int(duration*fs))


def rms(x):
    return np.sqrt(np.mean(x ** 2))


def read_audio(filename, target_fs=None):
    # x, fs = librosa.load(filename, sr=target_fs) # was producing error is a few datasets
    if '.flac' in filename[-5:]:
        # x, fs = sf.read(filename)
        x = AudioSegment.from_file(filename, format='flac')
        fs = x.frame_rate
        x = np.array(x.get_array_of_samples())

    elif '.wav' in filename[-5:]:
        fs, x = read(filename)
    else:
        raise ValueError('Error: Cannot load file' % filename)

    if np.isnan(x).any() or np.isinf(x).any():
        raise ValueError('Error: NaN or Inf value. File path: %s.' % filename)
    if not (isinstance(x[0], np.float32) or isinstance(x[0], np.float64)):
        x = x.astype('float32') / np.power(2, 15)
    if target_fs is not None and fs != target_fs:
        x = librosa.resample(x, orig_sr=fs, target_sr=target_fs)
        fs = target_fs
        # print('resampling...')
    return fs, x


def write_audio(filename, fs, wav):
    if isinstance(wav[0], np.float32) or isinstance(wav[0], np.float64):
        wav = np.asarray(np.multiply(wav, 32768.0), dtype=np.int16)
    write(filename, fs, wav)

def apply_raised_cosine_ramp(x, fs, onset_duration_ms=50, offset_duration_ms=50):
    ramp = np.ones(np.array(x).shape[-1])
    t = np.arange(np.pi*(3/2), np.pi*(5/2), np.pi/(round(onset_duration_ms * (fs / 1000))-1))
    ramp[:len(t)] = (np.sin(t) + 1) / 2
    t = np.arange(np.pi*(3/2), np.pi*(5/2), np.pi/(round(offset_duration_ms * (fs / 1000))-1))
    ramp[-len(t):] = np.flip((np.sin(t) + 1) / 2)
    if x.ndim is 1:
        return np.multiply(x, ramp)
    else:
        return np.multiply(np.array(x), ramp)


def show_progress(block_num, block_size, total_size):
    print('%.2f / 100%%' % (block_num * block_size / total_size * 100), end="\r")


# define the hook function
def hook(t):
    def inner(b=1, bsize=1, tsize=None):
        if tsize is not None:
            t.total = tsize
        t.update(b * bsize - t.n)
    return inner

