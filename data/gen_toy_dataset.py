import matplotlib.pyplot as plt
import numpy as np
from gen_signal import build_empty_signal, gauss_sig
from gen_noise_spectrum import generate_time_from_psd, f2

import torch
from torch.utils.data import TensorDataset, DataLoader

def build_toy_dataset(size = 10000, batch_size = 512):

    sr = 2**5
    # gaussian sig parameters
    
    std_min = 10    
    std_max = 30
    amp_min = 1
    amp_max = 10
    num_pulses = 1
    std_length = 3

    # generate psd first, assuming some binning(anything)
    largest_frequency = 1000
    num_bins = 700
    freqs = np.linspace(0, largest_frequency, num_bins)
    
    # generate spikes of noise
    num_spikes = 10
    psd = np.zeros(num_bins)

    X = []
    Y = []
    for i in range(size):
        psd, freq_spectrum, noise_time = generate_time_from_psd(num_bins*2, sr, largest_frequency, num_bins, f2, {"num_spikes":10})

        x, t = build_empty_signal(noise_time.shape[0], sr)
        sig = gauss_sig(x, t, std_min, std_max, amp_min, amp_max, num_pulses, std_length)
        X.append(sig + noise_time)
        Y.append(noise_time)
        
    X, Y = torch.tensor(X), torch.tensor(Y)
    dataset = TensorDataset(X, Y)
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    X = []
    Y = []
    for i in range(size):
        psd, freq_spectrum, noise_time = generate_time_from_psd(num_bins*2, sr, largest_frequency, num_bins, f2, {"num_spikes":10})
    
        x, t = build_empty_signal(noise_time.shape[0], sr)
        sig = gauss_sig(x, t, std_min, std_max, amp_min, amp_max, num_pulses, std_length)
    
        X.append(sig + noise_time)
        Y.append(noise_time)
            
    X, Y = torch.tensor(X), torch.tensor(Y)
    dataset = TensorDataset(X, Y)
    test_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return train_dataloader, test_dataloader