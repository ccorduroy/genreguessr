Caitlin Sullivan <ccsulliv@usc.edu>

# Repository Inventory

This document will walk you through the components of this project and describe what
each one does and how to run the model if you so choose.

## GTZAN Dataset
Original: ([Kaggle](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification))

### [gtzan local copy](gtzan/)

Should be cleaned. Includes feature normalization included. 

### [dataset_health_check.py](dataset_health_check.py)

Checks for corrupted files and quarantines them in [quarantined_data](quarantined_data/). Run this if you're using a fresh download.

Does NOT remove rows from feature .csv files. You'll need to do this elsewhere.

## Generating Spectrograms

### [spectrogram_generator.py](spectrogram_generator.py)

A modified version of its parent program, [Spectrogram_Generator_Source.ipynb](Spectrogram_Generator_Source.ipynb)
(Authored by Le and adapted)

Generate spectrograms of a set duration and size for the GTZAN dataset. You can just run this
script with whatever method you want uncommented at the bottom. Comment it back out when finished.

Options to limit to specific genres/parent files or singles. See comments at the bottom 
of the script.

Spectrograms are saved to a directory in your home directory named with the length of each
in seconds and an apt descriptor. The models are hooked up to recognize the name of this file, 
so don't change it. 

`window_duration`: Argument to the generator functions. sets the length (in time) of each 
individual spectrogram. Will split each .wav file from the dataset into spectrograms of 
this specified length. 

`HOP_LENGTH`: a constant calculated to make the spectrograms relatively square. Defines how
many of the time-sampled STFT bins to skip between columns graphed on the spectrogram.

*The 10-second CNN has its own deprecated copy of the generator that used an old compression
method. This is still present because that model was trained using that old method. Use that
script to generate spectrograms ONLY for the model in the same directory. 

## Explorations

### [model exploration pipeline (self-contained)](pipeline/)

Run main.py to do the exploration again. Graphs are already generated and in a subdirectory, 
so there's no need. 

Additional disjoint explorations in [model_exploration](model_exploration/)

## Trained CNNs

### [V3 - MiniVGG 10-second Spectrograms 2 VGG blocks (59% acc)](CNN/test_CNN_v3.ipynb)

PyTorch model on [Google Drive](https://drive.google.com/file/d/1601cNrAf7GmUEvB8PuA6o_q75wxHrsq2/view?usp=sharing)
(no indices :( )

### [V4 - MiniVGG 10-second Spectrograms 3 VGG blocks (Failed - re-scrambled dataset split when tested)](CNN/CNN_v4_10sec_trained/)

PyTorch model on [Google Drive](https://drive.google.com/file/d/1_mKhaywW2szC2p2WndR7mhWT63rcF4vV/view?usp=sharing)

### [V5 - MiniVGG 15-second Spectrograms 3 VGG blocks](CNN/test_CNN_v5_15_sec.ipynb)

PyTorch model on [Google Drive](https://drive.google.com/file/d/1EIHywgUoRt2RBCf7Lm48tnj_GCotLJkt/view?usp=sharing)

### [V7 - ResNet34 (88% acc) (used in live demo)](CNN/CNN_v7_3sec_resNet34_trained/)

PyTorch model on [Google Drive](https://drive.google.com/file/d/1eCEd4OZKG0cxTSgLtyc0yAY4KobI2mK7/view?usp=sharing)

### [V9 - ResNet34 with Appended Features (Failed - not normalized) (53% acc)](CNN/CNN_v9_3_sec_resnet34_with_features_trained/)

PyTorch model on [Google Drive](https://drive.google.com/drive/folders/1WjDy1N1OjULqLlMynVeEZ3TB-CxSaKXA?usp=sharing)

### [V10 - ResNet34 with Appended Features (92% acc)](CNN/CNN_v10_3_sec_resnet34_with_features_trained/)

PyTorch model on [Google Drive](https://drive.google.com/drive/folders/1fOcC0cJIbs21rkJkSn8TwJSQs_Z7BOfQ?usp=sharing)

## Live Recording

### [live_sample_denoised.py](live_sample_denoised.py)

Records audio, generates sample spectrograms to show denoising, and generates the spectrogram
expected by the CNN. CNN-ready spectrogram as well as the raw and denoised .wav files are saved
to the main directory (this script is only meant for a single demo).

Taking a new recording will overwrite previous one unless it's moved out of the top directory.