import numpy as np
import matplotlib.pyplot as plt
# from scipy.fft import fft
from scipy.fftpack import fft
from scipy import stats

theme = { 'red'   : '#e41a1c'
        , 'blue'  : '#265285'
        , 'green': '#4daf4a'
        , 'pink' : '#984ea3'
        , 'purple' : '#6e0178'
        , 'orange': '#ff7f00'
        }

def fourier_transform(actions, T):
    N = len(actions)
    yf = fft(actions)
    freq = np.linspace(0.0, 1.0/(2*T), N//2)
    amplitudes = 2.0/N * np.abs(yf[0:N//2])
    return freq, amplitudes

def smoothness(amplitudes):
    normalized_freqs = np.linspace(0, 1, amplitudes.shape[0])
    return np.mean(amplitudes * normalized_freqs)

def center_of_mass(freqs, amplitudes):
    return np.sum(freqs * amplitudes) / sum(amplitudes)

def cut_data(actionss, ep_lens):
    median = int(np.median(ep_lens))
    print("median:", median)
    same_len = map(lambda x: x[:median], filter(lambda x: len(x) >= median, actionss))
    return same_len

def to_array_truncate(l):
    min_len = min(map(len, l))
    return np.array(list(map(lambda x: x[min_len:], l)))


def combine(fouriers):
    freqs = fouriers[0][0]
    amplitudess = np.array(list(map(lambda x: x[1], fouriers)))

    amplitudes = np.mean(amplitudess, axis=0)
    return freqs, amplitudes

def from_actions(actionss, ep_lens):
    fouriers = list(map(fourier_transform, cut_data(actionss, ep_lens)))
    return combine(fouriers)

def plot_fourier(ax_m, freqs, amplitudes, amplitudes_std=None, main_color=theme['blue'], std_color=theme['orange']):
    if not (amplitudes_std is None):
        y = amplitudes + amplitudes_std
        ax_m.fill_between(freqs, 0, y, where=y >= 0, facecolor=std_color, alpha=1)
    ax_m.fill_between(freqs, 0, amplitudes, where=amplitudes >= 0, facecolor=main_color)

def dict_elems_apply(fn, d):
	return {k: fn(d[k]) for k in d.keys()}

def dicts_list_to_list_dicts(l):
	return {k: [dic[k] for dic in l] for k in l[0]}