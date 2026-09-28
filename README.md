# Urban Sound Classification and Salience Analysis

### Environmental audio classification with YAMNet embeddings and a spectrogram CNN

A machine-learning project exploring environmental audio using the **UrbanSound8K** dataset.

The project investigates two related tasks:

- classifying audio into ten environmental sound categories using pretrained **YAMNet** representations;
- predicting whether a sound is foreground or background using a custom convolutional neural network trained on mel spectrograms.

A final experiment tests whether predicted salience information improves sound-source classification.

## Results

| System | Task | Held-out accuracy |
| --- | --- | ---: |
| YAMNet + Logistic Regression | 10-class sound classification | **69.19%** |
| Spectrogram CNN | Foreground/background classification | **64.00%** |
| YAMNet + predicted salience | 10-class sound classification | **68.32%** |

The additional salience feature did **not** improve the YAMNet baseline: classification accuracy decreased from 69.19% to 68.32%.

For the salience model, the final decision threshold was selected using validation macro-F1 rather than assuming a fixed probability threshold of 0.5.

## Dataset and evaluation

The project uses **UrbanSound8K**, which contains 8,732 labelled urban sound clips across ten classes.

To keep the experiment computationally manageable while preserving a clean held-out evaluation:

- folds 1 and 2 are used for development;
- fold 3 is held out for final testing;
- training and test audio paths are explicitly checked for overlap.

This gives:

| Split | Clips |
| --- | ---: |
| Training/development | 1,761 |
| Held-out test | 925 |

The ten sound classes are:

- air conditioner
- car horn
- children playing
- dog bark
- drilling
- engine idling
- gun shot
- jackhammer
- siren
- street music

## Experiment 1: sound-source classification

A pretrained **YAMNet** model is used as a frozen feature extractor.

Each audio clip is:

1. loaded as mono audio at 16 kHz;
2. passed through YAMNet;
3. represented by the mean of its time-dependent 1,024-dimensional embeddings;
4. classified using class-balanced logistic regression.

The held-out fold reached **69.19% accuracy**.

Performance varied considerably by class. Gun shots, car horns, drilling, children playing, sirens and dog barks were recognised comparatively well, while air conditioners, engine idling and jackhammers were more difficult to distinguish.

## Experiment 2: salience classification

UrbanSound8K also labels recordings according to whether the target sound is in the foreground or background.

For this task, audio is converted into **64-bin mel spectrograms** and passed to a custom CNN consisting of:

- three convolutional blocks;
- batch normalisation;
- max pooling;
- global average pooling;
- a dense layer with dropout;
- a sigmoid binary-classification output.

Class weights address the imbalance between foreground and background recordings.

The probability threshold is selected using validation macro-F1 before evaluating once on the held-out test fold.

The final test accuracy was **64.00%**.

## Combined experiment

The final experiment asks whether predicted foreground/background salience adds useful information to YAMNet embeddings.

The CNN-predicted salience value is appended as an additional feature to the YAMNet representation before fitting logistic regression.

The combined system reached **68.32% accuracy**, slightly below the original YAMNet baseline of 69.19%.

This negative result is useful: under this setup, explicitly adding the predicted salience signal did not improve environmental sound classification.

## Repository structure

```text
audio-music-analysis/
├── audio_music_analysis.ipynb
├── README.md
├── requirements.txt
└── .gitignore

## Tech

**Python · TensorFlow · Keras · TensorFlow Hub · YAMNet · librosa · scikit-learn · NumPy · pandas · Matplotlib · audio signal processing · mel spectrograms · convolutional neural networks · transfer learning · logistic regression · classification · error analysis**
