# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 15:58:00 2019

@author: a-kojima
"""
import numpy as np
from scipy.fftpack import fft
from . import util

class minimum_variance_distortioless_response:
    
    def __init__(self,
                 mic_angle_vector,
                 mic_diameter,
                 sampling_frequency=16000,
                 fft_length=512,
                 fft_shift=256,
                 sound_speed=343):    
        self.mic_angle_vector=mic_angle_vector
        self.mic_diameter=mic_diameter
        self.sampling_frequency=sampling_frequency
        self.fft_length=fft_length
        self.fft_shift=fft_shift
        self.sound_speed=sound_speed
    
    def get_sterring_vector(self, look_direction):
        number_of_mic = len(self.mic_angle_vector)
        frequency_vector = np.linspace(0, self.sampling_frequency, self.fft_length)
        steering_vector = np.ones((len(frequency_vector), number_of_mic), dtype=np.complex64)
        look_direction = look_direction * (-1)
        for f, frequency in enumerate(frequency_vector):
            for m, mic_angle in enumerate(self.mic_angle_vector):
                steering_vector[f, m] = np.complex(np.exp(( - 1j) * ((2 * np.pi * frequency) / self.sound_speed) \
                               * (self.mic_diameter / 2) \
                               * np.cos(np.deg2rad(look_direction) - np.deg2rad(mic_angle))))
        steering_vector = np.conjugate(steering_vector).T
        normalize_steering_vector = self.normalize(steering_vector)
        return normalize_steering_vector[0:np.int(self.fft_length / 2) + 1, :]    
    
    def normalize(self, steering_vector):
        for ii in range(0, self.fft_length):            
            weight = np.matmul(np.conjugate(steering_vector[:, ii]).T, steering_vector[:, ii])
            steering_vector[:, ii] = (steering_vector[:, ii] / weight) 
        return steering_vector        
    
    def get_spatial_correlation_matrix(self, multi_signal, use_number_of_frames_init=10, use_number_of_frames_final=10):
        # init
        number_of_mic = len(self.mic_angle_vector)
        frequency_grid = np.linspace(0, self.sampling_frequency, self.fft_length)
        frequency_grid = frequency_grid[0:np.int(self.fft_length / 2) + 1]
        start_index = 0
        end_index = start_index + self.fft_length
        speech_length, number_of_channels = np.shape(multi_signal)
        R_mean = np.zeros((number_of_mic, number_of_mic, len(frequency_grid)), dtype=np.complex64)
        used_number_of_frames = 0
        
        # forward
        for _ in range(0, use_number_of_frames_init):
            multi_signal_cut = multi_signal[start_index:end_index, :]
            complex_signal = fft(multi_signal_cut, n=self.fft_length, axis=0)
            for f in range(0, len(frequency_grid)):
                    R_mean[:, :, f] = R_mean[:, :, f] + \
                        np.multiply.outer(complex_signal[f, :], np.conj(complex_signal[f, :]).T)
            used_number_of_frames = used_number_of_frames + 1
            start_index = start_index + self.fft_shift
            end_index = end_index + self.fft_shift
            if speech_length <= start_index or speech_length <= end_index:
                used_number_of_frames = used_number_of_frames - 1
                break            
        
        # backward
        end_index = speech_length
        start_index = end_index - self.fft_length
        for _ in range(0, use_number_of_frames_final):
            multi_signal_cut = multi_signal[start_index:end_index, :]
            complex_signal = fft(multi_signal_cut, n=self.fft_length, axis=0)
            for f in range(0, len(frequency_grid)):
                R_mean[:, :, f] = R_mean[:, :, f] + \
                    np.multiply.outer(complex_signal[f, :], np.conj(complex_signal[f, :]).T)
            used_number_of_frames = used_number_of_frames + 1
            start_index = start_index - self.fft_shift
            end_index = end_index - self.fft_shift            
            if  start_index < 1 or end_index < 1:
                used_number_of_frames = used_number_of_frames - 1
                break                    

        return R_mean / used_number_of_frames  
    
    def get_mvdr_beamformer(self, steering_vector, R):
        number_of_mic = len(self.mic_angle_vector)
        frequency_grid = np.linspace(0, self.sampling_frequency, self.fft_length)
        frequency_grid = frequency_grid[0:np.int(self.fft_length / 2) + 1]        
        beamformer = np.ones((number_of_mic, len(frequency_grid)), dtype=np.complex64)
        for f in range(0, len(frequency_grid)):
            R_cut = np.reshape(R[:, :, f], [number_of_mic, number_of_mic])
            inv_R = np.linalg.pinv(R_cut)
            a = np.matmul(np.conjugate(steering_vector[:, f]).T, inv_R)
            b = np.matmul(a, steering_vector[:, f])
            b = np.reshape(b, [1, 1])
            beamformer[:, f] = np.matmul(inv_R, steering_vector[:, f]) / b # number_of_mic *1   = number_of_mic *1 vector/scalar        
        return beamformer
    
    def apply_beamformer(self, beamformer, complex_spectrum):
        number_of_channels, number_of_frames, number_of_bins = np.shape(complex_spectrum)        
        enhanced_spectrum = np.zeros((number_of_frames, number_of_bins), dtype=np.complex64)
        for f in range(0, number_of_bins):
            enhanced_spectrum[:, f] = np.matmul(np.conjugate(beamformer[:, f]).T, complex_spectrum[:, :, f])
        return util.spec2wav(enhanced_spectrum, self.sampling_frequency, self.fft_length, self.fft_length, self.fft_shift)        