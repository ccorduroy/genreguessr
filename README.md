Le Duong <lnduong@usc.edu>

Samrit Grover <ssgrover@usc.edu>

Evangelos Neophytou <neophyto@usc.edu>

Caitlin Sullivan <ccsulliv@usc.edu>

# EE 460 Final Project: Music Genre Guesser

### [goto: Repository Inventory](INVENTORY.md)

## 1. Introduction

### 1.1: Motivation

Music genre classification is a fundamental task in the field of music information 
retrieval, with applications ranging from recommendation systems, Shazam, and audio 
indexing to music therapy and automated DJing. While traditional approaches rely on 
handcrafted features or metadata, they often fall short when dealing with raw audio 
content. In this project, we aim to improve genre classification by training a deep 
learning model on 10-second audio clips from GTZAN dataset. Our approach combines CNNs, 
which are well-suited for extracting spatial features from spectrograms, with RNNs 
which effectively model temporal dependencies in sequential audio data. This hybrid 
architecture allows the model to capture both timbral textures and rhythmic patterns 
that are important for distinguishing between musical genres. We also validate the 
model’s performance using spectrograms we generate from new sound samples allowing 
real-world testing beyond the original dataset.

### 1.2: Summary of Existing Approaches

Computer vision methods on spectrograms:
- Perform FFT on wav files (time domain) and analyze song in frequency domain. 
- Using an overlapping sample sliding window (~10ms) to create spectrograms.
- Spectrograms are then fed into CNN.

Training MLPs on known numerical features (GTZAN 60-feature, Spotify API features):
- Spotify API uses abstract features that are combinations of frequency data. Some repos on GitHub tried running FNNs on this set of features and got very low accuracy compared to spectrogram CV.
- Codes on Kaggle under GTZAN have tried most linear models on the dataset with varying outcomes.


### 1.3: Anticipated Results and Milestones

1. It is expected that a CNN+RNN model would have higher accuracy compared to CNN alone. A CNN
in general will have better performance than non deep-learning models.
2. Appending numerical features to spectrograms would increase the accuracy. 
3. We will train multiple models (PCA+Random Forest, RBF+FNN) to compare with CNN model.
4 We will explore various spectrogram lengths (different Nyquist frequencies & fft bins, 
different time length).

---

## 2. Problem Formulation

### 2.1: Data Strategy (Collecting, Cleaning, Augmenting)

There are 10 different labels which are based on the music genre: `blues`, `classical`, 
`country`, `disco`, `hiphop`, `jazz`, `metal`, `pop`, `reggae`, and `rock`. For each 
music genre, there are 100 audio files of length 30 seconds. 

For each audio file there are 59 
features derived including `filename`, `length`, `chroma_stft_mean / var`, `RMS_mean / var`, 
`spectral_centroid_mean / var`, `spectral_bandwidth_mean / var`, `rolloff_mean / var`, 
`zero_crossing_rate_mean / var`, `harmony_mean / var`, `perceptr_mean / var`, `tempo`, 
and 20 `mfcc_mean / var`. There are no missing values in any of the data columns, so no imputation 
is required. We removed the filename and length column since they’re not indicative 
of the song genre. For each of the features, we performed Min-Max normalization to 
scale the values between 0 and 1.

Some of the .wav files are corrupted (one in our download, to be exact). 
[dataset_health_check.py](dataset_health_check.py) examines all .wav files for validity 
and quarantines corrupted files so they are not used in the train/test split. 

### 2.2: Model Exploration

All combinations of the following data transformations and models were tested:

```
transformations = {
    "none": no_transform,
    "pca": pca_transform,
    "rbf_kernel" : rbf_transform,
    "polynomial_kernel" : polynomial_kernel
}

models = {
    "logistic_regression": train_logistic,
    "svm": train_svm,
    "perceptron": train_perceptron,
    "ridge": train_ridge,
    "decision_tree": train_decision_tree,
    "random_forest": train_random_forest,
    "KNN": train_knn,
    "neural_net" : train_nn
}
```

#### Model Performance Summary

![summary](./model_exploration/plots/accuracy_table.png)

#### All Loss and Accuracy Functions: See [model exploration plots](model_exploration/plots)
Note trial refers to one iteration of hyperparameter tuning. 


#### Transformation Key Insights
#### ↪ Polynomial Kernel Transformation:
- Best performer overall for linear models (logistic regression, ridge, SVM).
- Likely expands the feature space in a way that linear models can leverage non-linearly separable boundaries.
- Works surprisingly well even for neural nets and KNN.

#### ↪ RBF Kernel Transformation:
- Performs consistently poorly for most models (especially perceptron and neural net).
- RBF might distort the feature space too much, especially without proper tuning of hyperparameters (like gamma) or standardization.
- Could lead to overfitting on training or loss of meaningful structure.

#### ↪ No Transformation:
- Still strong for tree-based methods (e.g., random forest, decision trees) and KNN.
- These models work well with original feature distributions when the raw features are informative.

#### ↪ PCA Transformation:
- PCA slightly reduces performance across the board.
- Likely due to dimensionality reduction discarding useful information for classification.
- Exception: helps reduce overfitting or redundancy for neural nets and random forests, but not significantly here.

#### Specific Model Behavior
- KNN performs best overall with raw data and polynomial kernel. Likely due to high-dimensional, localized patterns in GTZAN features.
- Random Forest is also robust across all transforms — not sensitive to feature scaling.
- Neural Network performs poorly on RBF, but excels otherwise. Likely affected by feature scale distortion from RBF.
- Linear models (Logistic, Ridge, SVM) are greatly boosted by the polynomial kernel, supporting the idea that the original data isn’t linearly separable.

#### Conclusions
- Use polynomial kernel transformation when working with linear models.
- Avoid RBF transformation unless you tune hyperparameters carefully.
- KNN and Random Forest remain the most robust across transformations.
- Neural networks benefit from more structured expansions like polynomial features, not arbitrary warping like RBF.


### 2.3: Main Model Choice and Training Approaches

### 2.4: Final System Architecture

### 2.5: Real-World Inference

[live_sample_denoised.py](live_sample_denoised.py) records 10 seconds of audio and uses
**spectral subtraction** to remove static background noise. This method uses a reference 
sample of just noise, analyzes the frequency-domain and power aspects of just the noise,
then subtracts them from the music sample while retaining phase.

#### Spectral Subtraction

1. **Noise estimation**

    - A noise-only segment taken before the music sample is used to estimate 
    the average noise spectrum, $N(f)$
    - This is done by computing the 
    Short-Time Fourier Transform (STFT) on the noise sample.


2. **STFT of noisy signal only**

    - The noisy music signal is split into overlapping frames using a window function.

    - STFT is applied to get the time-frequency representation $Y(f,t)$, where $f$ is 
    frequency and $t$ is time.


3. **magnitude subtraction from music sample**

    - For each frame and frequency bin, the estimated noise magnitude spectrum is \
   subtracted:

        $∣X(f,t)∣=max(∣Y(f,t)∣−∣N(f)∣,\epsilon)$
   
    - where $\epsilon$ is a small constant to avoid negative values


4. **phase retention**

    - The phase from the original noisy signal is retained:
   
        $X(f,t) = ∣X(f,t)∣ *  e^{j∠Y(f,t)}$


5. **reconstruction**

    - Inverse STFT (ISTFT) is used to convert the modified spectrum back into the time 
   domain, reconstructing the denoised signal.

This method does not require training data or complex models like high-end denoising and is
effective for stationary noise, like AC humming or electrical buzz. It retains musical
signature well because it only subtracts magnitude and keeps phase. That being said, it
requires a good noise estimate and assumes noise is the same through the whole sample. 
Parameters like noise floor, spectral flooring, and smoothing can be tunes for different
genres or instruments as well, but that tuning isn't used here. \

#### Spectral Subtraction Example

![Noisy](./writeup_images/live_audio_spectrogram.png "Raw Sample Spectrogram")

![Noise Estimate](./writeup_images/estimated_noise_spectrum.png "Noise Estimate Spectrogram")

![Denoised](./writeup_images/denoised_live_audio_spectrogram.png "Denoised Sample")

#### Real-World Inference

This denoised sample is turned into a spectrogram of the size expected by the CNN and can then
be fed in for inference. 

---

## 3. Outcomes

### 3.1: CNN Models Performance on Train/Val/Test

2 types of architectures were used: Mini-VGG and ResNet34

Sources:
1) [Mini-VGG: modified from lecture](https://github.com/keithchugg/ee460_spring2025/blob/5b3d0470705ff236a212415c1e510630627afd2d/lecture/02_fmnist_mlp_torch.ipynb)
2) [ResNet34: modified from Kaggle](https://www.kaggle.com/code/khoongweihao/resnet-34-pytorch-starter-kit)

#### [V3: 15-Second Spectrogram Trained Mini-VGG (2-blocks)](https://github.com/ccorduroy/genreguessr/blob/de9a36fd135d69d0442dfa62df09a4e7ef80da44/CNN/test_CNN_v3.ipynb)

This is our first "solid" training for the Mini-VGG model. 

Our goal with this model is to test whether the Mini-VGG model is sufficient for what we want to do. However, can shown below, the model overfits, and the accurcy isn't very high. We suspected that the spectrograms' length could controbute to this issue, since 15 seconds is very dense and the convolution layers may not be able to pick up on the patterns. Additionally, since our original 30 seconds dataset only has 1000 data poins, this means our entire dataet for 15 seconds, which is pretty small for a spectrogram CNN (compared to the simple MNIST, which has 70,000 samples). Finally, we also suspec **TO BE CONTINUED

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1         [-1, 32, 480, 640]             896
              ReLU-2         [-1, 32, 480, 640]               0
       BatchNorm2d-3         [-1, 32, 480, 640]              64
            Conv2d-4         [-1, 32, 480, 640]           9,248
              ReLU-5         [-1, 32, 480, 640]               0
       BatchNorm2d-6         [-1, 32, 480, 640]              64
         MaxPool2d-7         [-1, 32, 240, 320]               0
         Dropout2d-8         [-1, 32, 240, 320]               0
            Conv2d-9         [-1, 64, 240, 320]          18,496
             ReLU-10         [-1, 64, 240, 320]               0
      BatchNorm2d-11         [-1, 64, 240, 320]             128
           Conv2d-12         [-1, 64, 240, 320]          36,928
             ReLU-13         [-1, 64, 240, 320]               0
      BatchNorm2d-14         [-1, 64, 240, 320]             128
        MaxPool2d-15         [-1, 64, 120, 160]               0
        Dropout2d-16         [-1, 64, 120, 160]               0
          Flatten-17              [-1, 1228800]               0
           Linear-18                  [-1, 512]     629,146,112
             ReLU-19                  [-1, 512]               0
      BatchNorm1d-20                  [-1, 512]           1,024
        Dropout1d-21                  [-1, 512]               0
           Linear-22                   [-1, 10]           5,130
================================================================
Total params: 629,218,218
Trainable params: 629,218,218
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 3.52
Forward/backward pass size (MB): 740.64
Params size (MB): 2400.28
Estimated Total Size (MB): 3144.43
----------------------------------------------------------------

Val: 100%|██████████| 4/4 [00:13<00:00,  3.29s/batch, loss=1.19, acc=59.9] 
Test accuracy: 0.5990

```

![v3_acc](https://github.com/user-attachments/assets/cac21d45-ca87-498e-ad6a-32b4e13ca3da)
![v3_loss](https://github.com/user-attachments/assets/d3839a46-3580-4033-8d7b-9a1b6be6ca31)

#### V4: 10-Second Spectrogram Trained VGG

Validation Accuracy (last epoch): 78.9%

![confusion matrix](CNN/CNN_v4_10sec_trained/confusionmatrix.png)


#### [V5: 15-second Spectrogram Mini-VGG (3-blocks)](https://github.com/ccorduroy/genreguessr/blob/837f3f954a09e46376bf6a565d2d06fc632723ff/CNN/CNN_v9_3_sec_resNet34_w_features.py)

#### V6: 3-second Spectrogram Trained VGG

`Train Eval: 100%|██████████| 219/219 [08:44<00:00,  2.39s/batch, loss=0.129, acc=99.2]`

`Val: 100%|██████████| 47/47 [01:53<00:00,  2.42s/batch, loss=0.727, acc=76.2]`

`Test: 100%|██████████| 47/47 [02:00<00:00,  2.55s/batch, loss=0.639, acc=78.7]`

This model overfit the train set. This might have been a result of the length of the 
spectrogram or the depth of the neural network. 

#### V7: 3-Second Spectrogram Trained ResNet


#### (BEST) V10: 3-Second Spectrogram Traines Resnet with Appended Numeric Features (Re-Normalized)

`Train: 100%|██████████| 219/219 [01:42<00:00,  2.14batch/s, loss=0.587, acc=77.4]`

`Train Eval: 100%|██████████| 219/219 [00:46<00:00,  4.70batch/s, loss=0.0136, acc=99.9]`

`Val: 100%|██████████| 47/47 [00:09<00:00,  4.74batch/s, loss=0.251, acc=91.6]`

`Val: 100%|██████████| 47/47 [00:11<00:00,  4.22batch/s, loss=0.252, acc=91.7]`

`Best Model Test accuracy: 0.9167`

![confusion matrix](CNN/CNN_v10_3sec_resnet34_with_features_trained/confusion_matrix.png)

<details>
<summary>Loss and Accuracy graphs per epoch (titles are wrong but data is correct)</summary>

![loss](CNN/CNN_v10_3sec_resnet34_with_features_trained/loss.png)

![acc](CNN/CNN_v10_3sec_resnet34_with_features_trained/acc.png)


</details>

### 3.2: Problems and Challenges

#### CNN Size and Computing Limits
Each CNN training (did 7) took almost an entire day on one device. On GPU, the
CNNs processing larger images took 14 hours or so each. It's probable that either the 
CNN needed longer spectrograms to make more accurate predictions or needed to be deeper--
both of these options would have exponentially increased model complexity outside of this
project's time frame and our computing capability.

Overfitting may also have been caused by dataset size compared to the complexity of the CNN. 
Since there are only 1,000 songs in the dataset, that results in only 3,000 data points if taking
10-second spectrograms or 10,000 if taking 3-second spectrograms. The latter is more sufficient
in size but each datapoint has much less information in it which could reduse efficiency. It would
be ideal to have more training data and more computing time regardless of architecture, but the
limitations are seen especially in the CNN.

### 3.3: Future Work

#### CNN Architecture Improvement
We would like to test more CNN architectures, probably with more VGG blocks or other Resnets,
to see if the accuracy of the CV method can be improved to be equal to or better than the KNN
running on the features provided by the dataset. This may have to be done by brute force by
training more models.

#### RNN Inclusion
We also were unable to implement the CNN-RNN fusion architecture planned for in the original 
outline, so future work would include this. The output vectors from the CNN would serve as
inputs to an RNN which could, theoretically, support inferences based off longer time samples,
which may improve accuracy for the spectrogram CV approach. The input to the RNN could also be
fed the mathematical features from the original dataset as the MLP layer of the CNN was in our
current implementation. The reason RNN would be beneficial is because Music is temporal, you can 
use both local and global information and the RNN might learn genre transitions 
(e.g., a song starts classical, becomes electronic). A sample draft for the RNN implementation
is found in rnn.ipynb.

#### Streamlined Live Inference
The ability to make an inference from continuous live recording instead of taking a single 
x-second sample would make the model more real-world applicable. An RNN would be useful in 
updating inference over time as a song plays for longer.

#### Data Augmentation
In order to improve the performance of the CNN, increase the dataset size and make it more 
applicable to real-world data (and more accurate on clean, simulation data) by adding
artificially deprecated versions of each audio sample to the dataset. This would increase 
the dataset size and allow the CNN to recognize microphone noise and compression.

### 3.4: Conclusions & What We Learned
- The CNN on 10-second spectrograms achieved a test accuracy of ~79%, outperforming many traditional machine learning approaches.
- Our CNN-based models successfully classified songs into 10 genres using spectrograms from the GTZAN dataset.
- The hybrid CNN+RNN architecture (planned) is promising for capturing both spatial and temporal patterns in audio, though not fully implemented yet.
- The Linear Model (KNN) on 10-second audiofiles achieved a test accuracy of ~93%, outperforming many traditional machine learning approaches.
- The project highlighted the importance of feature quality, data augmentation, and architecture choice in music genre classification tasks.
- Our live audio inference pipeline worked effectively, with real-world spectrograms being denoised, processed, and classified by the trained model.
- Learned the effectiveness and the usefulness of spectrograms
- Learned the difference between generalization of the model for personal use and generalization for the outside world.

---

## 4. Task Summary and Teamwork Breakdown

Le Duong: 

- Generated 7 CNNs to test architecture and input data dimension
- CNN training and inference pipeline
- Wrote original spectrogram generator protocol
- developed method to concatenate features onto last layer of CNN


Samrit Grover: 

- 4x6 model exploration
- Result reports and interpretation


Evangelos Neophytou: 

- Model exploration: code, graphs, neural network imp
- RNN implementation (not used in final model but present in project)
- PCA and K-means explorations
- Presentation


Caitlin Sullivan: 

- Scripted data cleaning
- Live data collection pipeline: recording, de-noising, creating spectrogram, feeding to CNN
- Adjusted spectrograms: resizing, downsampling for different CNN architectures
- CNN training and inference pipeline
- Report
