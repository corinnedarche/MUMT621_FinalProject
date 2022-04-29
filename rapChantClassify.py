#!/usr/bin/env python
# coding: utf-8

# SCRIPT: rapChantClassify.py
# CREATED BY: Corinne Darche
# PURPOSE: Input a 10-second audio file of either Gregorian Chant or Rap, and run it through a pre-trained 
# SVM, GMM, and K-Means classifier to identify its genre

import numpy as np
import sys, getopt
import os
import librosa
import pickle
import joblib

from sklearn import svm
from sklearn.linear_model import SGDClassifier
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing


def vocal_separator(y, sr):
    # FUNCTION: Inspired by the "Vocal separator" in the librosa documentation, this separates vocals from their background
    # Part of preprocessing before features are extracted
    S_mag, S_phase = librosa.magphase(librosa.stft(y))
    S_filter = librosa.decompose.nn_filter(S_mag, 
                                           aggregate=np.median, 
                                           metric='cosine', 
                                           width=int(librosa.time_to_frames(2, sr=sr)))
    
    S_filter = np.minimum(S_mag, S_filter)
    margin_i, margin_v = 2, 10
    power = 2
    
    mask_i = librosa.util.softmask(S_filter, 
                                   margin_i * (S_mag - S_filter), 
                                   power=power)
    
    mask_v = librosa.util.softmask(S_mag - S_filter, 
                                   margin_v * S_filter, 
                                   power=power)
    
    S_foreground = mask_v * S_mag
    S_background = mask_i * S_mag
    
    y_foreground = librosa.istft(S_foreground * S_phase)
    
    return y_foreground

def feature_extraction(audio, sr):
    # FUNCTION: This extracts nine key features from an input audio file
    # Chroma
    chroma = np.mean(librosa.feature.chroma_stft(audio, sr=sr))
    # Spectral centroid
    centroid = np.mean(librosa.feature.spectral_centroid(audio, sr=sr))
    # Spectral bandwidth
    bandwidth = np.mean(librosa.feature.spectral_bandwidth(audio, sr=sr))
    # Spectral flatness
    flatness = np.mean(librosa.feature.spectral_flatness(audio))
    # Spectral rolloff
    rolloff = np.mean(librosa.feature.spectral_rolloff(audio, sr=sr))
    # MFCCs
    mfcc = np.mean(librosa.feature.mfcc(audio, sr=sr))
    # RMS
    rms = np.mean(librosa.feature.rms(y=audio))
    # Zero-crossing Rate
    zero = np.mean(librosa.feature.zero_crossing_rate(y=audio))
    # Tempogram (rhythm)
    tempogram = np.mean(librosa.feature.tempogram(y=audio, sr=sr))
    
    return [chroma, centroid, bandwidth, flatness, rolloff, mfcc, rms, tempogram, zero]


def main(argv):
    inputfile = ''
    genre = ''
    # Input for script
    try:
        opts, args = getopt.getopt(argv, "hi:g:", ["help", "ifile=", "genre="])
    except getopt.GetoptError as e:
        print(e)
        sys.exit(2)

    for opt, arg in opts:
        if opt in ['-h', "--help"]:
            print('\nusage: rapChantClassify.py -i inputFile -g genre')
            print('\tinputFile: path to a valid 10-second input file to classify')
            print("\tgenre: inputFile's genre. Only accepts rap and (Gregorian) chant.\n")
            sys.exit()
        elif opt in ["-i", "--ifile"]:
            if os.path.exists(arg):
                inputfile = arg
            else:
                print("Error: file/path invalid. Please input a valid file.")
                sys.exit()
        elif opt in ["-g", "--genre"]:
            if 'chant' == arg.lower() or 'rap' == arg.lower():
                genre = arg
            else:
                print("Error: This classifier script only works on (Gregorian) chant or rap genres.")
                sys.exit()
    
    # Input the file
    y, sr = librosa.load(inputfile, mono=True)
    yFilt = vocal_separator(y, sr)
    feats = feature_extraction(yFilt, sr)
    features = np.array(feats)

    # Load the classifiers
    svm = joblib.load('models/svm_classifier')
    gmm = joblib.load('models/gmm_classifier')
    
    svmResult = svm.predict([features])
    gmmRes = gmm.predict([features])

    gmmResult = ''

    if gmmRes[0] == 0:
        gmmResult = 'rap'
    else:
        gmmResult = 'chant'

    print('\n======= Classification Results =======\n')
    print('Genre to be classified: ' + genre)
    print('Support Vector Machine: ' + svmResult[0])
    print('Gaussian Mixture Model: ' + gmmResult)
    print()
    
if __name__ == "__main__":
    main(sys.argv[1:])