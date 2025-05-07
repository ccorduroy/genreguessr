Caitlin Sullivan <ccsulliv@usc.edu>

# Repository Inventory

This document will walk you through the components of this project and describe what
each one does and how to run the model if you so choose.

## GTZAN Dataset
Original: ([Kaggle](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification))

### [gtzan local copy](gtzan/)

Should be cleaned.

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

### [model exploration (self-contained)](model_exploration/)

Run main.py to do the exploration again. Graphs are already generated and in a subdirectory, 
so there's no need. 

### Others:

### [rnn.ipynb](rnn.ipynb)

RNN exploratory notebook. Didn't make the cut into the final architecture but is our proof-of-concept
for future work.

### [kmeansmodel.ipynb](./model_exploration/other/kmeansmodel.ipynb)

K-means clustering transformation exploration from early iterations.

### [pcamodel.ipynb](./model_exploration/other/pcamodel.ipynb)

Another PCA exploration from early iterations.

**note**: The other files inside model_exploration are Python copies of the Jupyter Notebooks.

## Trained CNNs

### [MiniVGG on 3-second Spectrograms (available)](CNN_3sec_trained/)

Trained PyTorch model on [Google Drive](https://drive.google.com/file/d/1Oh1phJA5a-hHz8WAXHOX1wS0f5qTMpUt/view?usp=sharing).

Dataset split indices and records on [Google Drive](https://drive.google.com/drive/folders/11s1Yt4oBUWH4NXrJmro4rryV5X_p_jPR?usp=sharing). 


The spectrogram generator is currently configured for this model. 

### [MiniVGG on 10-second Spectrograms (available)](CNN_10sec_trained/)

Trained PyTorch model on [Google Drive](https://drive.google.com/file/d/1_mKhaywW2szC2p2WndR7mhWT63rcF4vV/view?usp=sharing).

** when you're generating spectrograms for this model, use the python script in its directory.

**I think the results of this one were a fluke (reshuffled when testing). No valid indices saved.

### [ResNet-34 on 3-seocnd Spectrograms](CNN_v7_sec_resNet34/)

Another architecture. The spectrogram generator is currently configured for this model.

### [ResNet-34 on 3-seocnd Spectrograms with appended numeric features](CNN_v9_3sec_resnet34_appendedfeatures/)

Same as the other architecture but before training the spectrogram image data is augmented with 57 numeric features provided
by the GTZAN dataset csv.

## Model Loading/Inference

## Live Recording

### [live_sample_denoised.py](live_sample_denoised.py)

Records audio, generates sample spectrograms to show denoising, and generates the spectrogram
expected by the CNN. CNN-ready spectrogram as well as the raw and denoised .wav files are saved
to the main directory (this script is only meant for a single demo).

Taking a new recording will overwrite previous one unless it's moved out of the top directory.


