# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 11:49:02 2019

@author: a-kojima
"""
import numpy as np
import soundfile as sf
import matplotlib.pyplot as pl
from beamformer import complexGMM_mvdr as cgmm

SAMPLING_FREQUENCY = 16000
FFT_LENGTH = 512
FFT_SHIFT = 128
NUMBER_EM_ITERATION = 20
MIN_SEGMENT_DUR = 2
ENHANCED_WAV_NAME = './output/enhanced_speech_cgmm.wav'
IS_MASK_PLOT = True

def multi_channel_read(prefix=r'./sample_data/20G_20GO010I_STR.CH{}.wav',
                       channel_index_vector=np.array([1, 2, 3, 4, 5, 6])):
    wav, _ = sf.read(prefix.replace('{}', str(channel_index_vector[0])), dtype='float32')
    wav_multi = np.zeros((len(wav), len(channel_index_vector)), dtype=np.float32)
    wav_multi[:, 0] = wav
    for i in range(1, len(channel_index_vector)):
        wav_multi[:, i] = sf.read(prefix.replace('{}', str(channel_index_vector[i])), dtype='float32')[0]
    return wav_multi

multi_channels_data = multi_channel_read()

cgmm_beamformer = cgmm.complexGMM_mvdr(SAMPLING_FREQUENCY, FFT_LENGTH, FFT_SHIFT, NUMBER_EM_ITERATION, MIN_SEGMENT_DUR)

complex_spectrum, R_x, R_n, noise_mask, speech_mask = cgmm_beamformer.get_spatial_correlation_matrix(multi_channels_data)

beamformer, steering_vector = cgmm_beamformer.get_mvdr_beamformer(R_x, R_n)

enhanced_speech = cgmm_beamformer.apply_beamformer(beamformer, complex_spectrum)

sf.write(ENHANCED_WAV_NAME, enhanced_speech / np.max(np.abs(enhanced_speech)) * 0.65, SAMPLING_FREQUENCY)

if IS_MASK_PLOT:
    pl.figure()
    pl.subplot(2, 1, 1)
    pl.imshow(np.real(noise_mask).T, aspect='auto', origin='lower', cmap='hot')
    pl.title('noise mask')
    pl.subplot(2, 1, 2)
    pl.imshow(np.real(speech_mask).T, aspect='auto', origin='lower', cmap='hot')
    pl.title('speech mask')
    pl.show()