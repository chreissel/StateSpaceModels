import numpy as np
import matplotlib.pyplot as plt

# generate a power density spectrum of some function
def f(x): return x**2
    
def f2(x, num_spikes):
    x = 5*x**2
    for i in range(num_spikes):
        index = np.random.randint(0,len(x))
        magnitude = x[index] + np.random.randint(1000,100000)
        x[index] += magnitude
    return x

def generate_time_from_psd(n, sr, largest_frequency, num_bins, f, f_kwargs):
    
    freqs = np.linspace(0, largest_frequency, num_bins)
    psd = np.zeros(num_bins)
    
    psd += f(freqs, **f_kwargs)
    psd = psd/psd.sum()
    
    # plt.plot(psd)
    # plt.show()
    
    # convert to frequency spectrum
    freq_spectrum = np.sqrt(psd * sr * n)
    random_phase = np.exp(1j * np.random.uniform(0, 2 * np.pi, len(freq_spectrum)))
    freq_spectrum = random_phase * freq_spectrum
    
    # plt.figure(figsize=(20, 5))
    # plt.plot(np.real(freq_spectrum))
    # plt.plot(np.imag(freq_spectrum))
    # plt.show()
    
    # plt.figure(figsize=(20, 5))
    time_signal = np.fft.irfft(freq_spectrum)
    # plt.plot(time_signal)
    
    return psd, freq_spectrum, time_signal
