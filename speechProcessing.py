import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import IPython.display as ipd
from scipy.io import wavfile
import scipy.signal as sp
from scipy.fftpack import fft, ifft
import math
from os import listdir

# Setting up
plt.rcParams["figure.figsize"] = (14,4)
cutOff_freq = 2000
pathToRaw = './snd/rawDigits/'
pathToDigits = './snd/digits/'

# Signal Processing functions

# NOTE: Take out if don't use
def SNR(noisy, original):
    """Calculates the Signal to Noise Ratio in dB based from the signal and noise inputs."""
    return 10*np.log10((rms(original)/rms(noisy)))

def filterSignal(s, cutOff_freq, btype, sf):
    """Takes a signal and applies a Butterwoth filter to it. btype: 'lowpass’ or ‘highpass’"""
    nyquist = sf/2
    wc = cutOff_freq / nyquist
    b, a = sp.butter(6, wc, btype = btype)
    return sp.lfilter(b, a, s)

def rms(signal):
    """Returns the rms level of the signal."""
    sumofsquares = 0
    for v in signal:
        sumofsquares = sumofsquares + v**2
    mean_sumofsquares = sumofsquares / len(signal)
    return math.sqrt(mean_sumofsquares)

def filterSignal(s, cutOff_freq, btype, sf):
    """Takes a signal and applies a Butterwoth filter to it. btype: 'lowpass’ or ‘highpass’"""
    nyquist = sf/2
    wc = cutOff_freq / nyquist
    b, a = sp.butter(6, wc, btype = btype)
    return sp.lfilter(b, a, s)

def rms(signal):
    """Returns the rms level of the signal."""
    sumofsquares = 0
    for v in signal:
        sumofsquares = sumofsquares + v**2
    mean_sumofsquares = sumofsquares / len(signal)
    return math.sqrt(mean_sumofsquares)

def SSN(signals, sf):
    """Computes Speech Shaped Noise for a list of speech signals of the same length
    by randomizing the phases of the FFTs of each signal.
    Returns the speech shaped noise signal."""

    lmax = max([len(s) for s in signals])
    signals = [np.pad(s, (0, lmax-len(s)), mode = 'constant') for s in signals]
    signals = np.stack(signals)

    # Sum the FFT spectrums of each signal
    ss_fft = fft(signals).sum(axis = 0)

    # Randomize phase
    ss_fft = ss_fft * np.exp(1j * 2 * np.pi * np.random.rand(*ss_fft.shape))

    ssn_s = ifft(ss_fft).real
    ssn_s = filterSignal(ssn_s, 8000, 'lowpass', sf)
    return ssn_s


# Visualization Functions

def plot_fft(s, sf):
    """Computes the fft and shows its spectrum."""
    # number of samples
    N = len(s)
    # sampling spacing
    T = 1/sf

    # taken from https://docs.scipy.org/doc/scipy/reference/tutorial/fftpack.html
    yf = fft(s)
    xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
    plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.show()

# Main: Creating stimulus
# From raw wav recordings of each digit, get an array of signals of equal length and rewrite them to wav files
digits_files = listdir(pathToRaw)

# Make a dictionary wtih each digit recording: key = name of file, value = array of signal
digits = {}
for file in digits_files:
    sf, s = wavfile.read(pathToRaw + file)
    digits[file] = s

# Pad the signal arrays to make them all equal length
lmax = max([len(s) for s in digits.values()])
for file in digits:
    digits[file] = np.pad(digits[file], (0, lmax-len(digits[file])), mode = 'constant')

# Equalize the Root Mean Squared amplitude of the recordings
rms_max = max([rms(s) for s in digits.values()])
for file in digits:
    digits[file] = digits[file]/rms(digits[file]) * rms_max

# Write processed digit signals file
for file, snd in digits.items():
    snd.astype(float)
    wavfile.write(pathToDigits + 'post_' + file, sf, snd)

#Make a speech shaped noise out of them and write it to file
ssn = SSN(digits.values(), sf)
rms_ssn = rms(ssn)
wavfile.write('./snd/SSN/speech_shaped_noise.wav', sf, ssn)
plot_fft(ssn, sf)

# Creating all filter (x2) and SNR (x5) conditions for each digit
pathToStimuli = './snd/SIN/'
SNR_conds = [0, -3, -6, -9, -12]
freq_conds = ['lowpass', 'highpass']
hi_stim = np.empty((5,10,lmax))
lo_stim = np.empty_like(hi_stim)
# loop through each digit recording (x9)
for file, snd in digits.items():
    # Extract the digit from filename
    digit = int(file[0])

    # loop through conditions (x2)
    for i in range(len(freq_conds)):
        freq = freq_conds[i]
        s = filterSignal(snd, cutOff_freq, freq, sf = sf)

        # loop through Signal to Noise ration conditions (x5)
        for j in range(len(SNR_conds)):
            r = SNR_conds[j]
            # fixing rms level of signal relative to speech shaped noise
            x = 10**(-r/10)
            s = s/rms(s) * rms_ssn/x
            # Adding noise to signal
            stim = s + ssn

            wavfile.write(pathToStimuli + freq + '/' +  str(r) + '/' + file, sf, stim)

            if freq == 'lowpass':
                lo_stim[j][digit] = stim
            elif freq == 'highpass':
                hi_stim[j][digit] = stim




# Functions to generate stimulus for experiment
def generate_triplets(numTrials):
    """Generate digit triplets for numTrials number of trials.
    Each digit is presented the same number of times.
    Outputs an array of triplet arrays.
    To be generated for each participant."""
    n = numTrials*3/9
    digits = np.array(range(9)) + 1
    stimuli_set = np.repeat(digits, n)
    stimuli_rand = np.random.choice(stimuli_set, size = len(stimuli_set), replace = False)
    stimuli_triplets = stimuli_rand.reshape((-1, 3))
    return stimuli_triplets
