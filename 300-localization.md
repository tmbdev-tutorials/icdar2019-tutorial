```python
%pylab inline
```

```python
import torch
from torch import nn
from torchmore import layers, flex
```


# LOCALIZATION


# OCR Training Data

OCR training data usually consists of:

- an image of text
- a transcription of the text

We are usually not given:

- character bounding boxes
- word bounding boxes (when recognizing text lines)
- page segmentation (when recognizing whole pages)


# EM Algorithms

Expectation-Maximization is a general approach to learning when some variable we need for recognition is missing from the training data.

For OCR, the missing information is the _segmentation_ (e.g., character locations.

EM approach:
- make a first guess at the segmentation
- recognize assuming the segmentation is correct
- update the segmentation using the recognition output
- repeat


# CTC as an EM Algorithm

- perform scanning recognition (e.g. with LSTM)
- perform Viterbi/CTC alignment to find the best locations for each character
- use those locations as if they were ground truth
- compute cross entroy / MSE loss and backpropagate

CTC gives us horizontal positions of characters but no vertical locations.

What if we want XY positions for each character? Bounding boxes?


# Brute Force XY Character Positions

- input: unnormalized word images
- get X positions from CTC algorithm / DL output
- formally assign Y positions to characters as $\mu_y$ of the word image
- train a convolutional neural network using $(x_{CTC}, \mu_y)$ as the location for each character


# EM Algorithm for XY Character Positions

- input: unnormalized word images
- have a 2D convolutional network that outputs the probability of the presence of a character at each pixels $(x, y)$
- perform 2D beam search over positions to best match the transcribed word

# RCNN-like Algorithm

- you can implement region proposal algorithms directly for character bounding boxes
- the problem is complicated by the fact that there are frequently multiple instances of each character
- a direct implementation does not take advantage of the known left-to-right ordering of the transcript but simply treats transcripts as bags of characters


# Segmentation by Backpropagation / Masking

Several techniques in the literature have been developed to determine which regions in the input are responsible for a given output class:

- for each output location of a CTC-trained model, compute the derivative of the output with respect to the input pixels
- scan a mask across the input image and determine which mask locations affect which character classification

