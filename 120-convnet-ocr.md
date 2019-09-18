```python
%pylab inline
```

```python
import torch
from torch import nn
from torchmore import flex, layers
```


# APPLICATIONS TO OCR


# Character Recognition

- assuming you have a character segmentation
  - extract each character
  - feed to any of these architectures as if it were an object recognition problem

Goodfellow, Ian J., et al. "Multi-digit number recognition from street view imagery using deep convolutional neural networks." arXiv preprint arXiv:1312.6082 (2013).

# Word Recognition

- perform word localization using Faster RCNN
- perform whole word recognition as if it were a large object reconition problem

![word recognition](figs/word-recognition.png)

Jaderberg, Max, et al. "Deep structured output learning for unconstrained text recognition." arXiv preprint arXiv:1412.5903 (2014).


# Better Techniques

- above techniques are applications of computer vision localization
- Faster RCNN and similar techniques are ad hoc and limited
- often require pre-segmented text for training
- better approaches:
  - use markers for localizing/bounding text (later)
  - use sequence learning techniques and CTC for alignment and OCR learning


# Using Convolutional Networks for OCR

```python
def make_model():
    return nn.Sequential(
        # BDHW
        *convolutional_layers(),
        # BDHW, now reduce along the vertical
        layers.Fun(lambda x: x.sum(2)),
        # BDW
        layers.Conv1d(num_classes, 1)
    )
```


# Training Procedure for Convolutional Networks

- pass the input image (BDHW) through the model
- output is a sequence of vectors (BDW=BDL)
- perform CTC alignment between output sequence and text string
- compute the loss using the aligned output sequence
- backpropagate and update weights


# Viterbi Training

- ground truth: text string = sequence of classes
- ground truth `"ABC"` is replaced by regular expression `/_+A+_+B+_+C+_+/`
- network outputs $P(c|i)$, a probability of each class $c$ at each position $i$
- find the best possible alignment between network outputs and ground truth regex
- that alignment gives an output for each time step
- treat that alignment as if it were the ground truth and backpropagate
- this is an example of an EM algorithm


# CTC Training

- like Viterbi training, but instead of finding the best alignment uses an average alignment

Identical to traditional HMM training in speech recognition:

- Viterbi training = Viterbi training
- CTC training = forward-backward algorithm


# cctc2

- with the `cctc2` library, we can make the alignment explicit
```python
def train_batch(input, target):
    optimizer.zero_grad()
    output = model(input)
    aligned = cctc2.align(output, target)
    loss = mse_loss(aligned, output)
    loss.backward()
    optimizer.step()
```


# CTC in PyTorch

- in PyTorch, CTC is implemented as a loss function
- `CTCLoss` in PyTorch obscures what's going on
- all you get is the loss output, not the EM alignment
- sequences are packed in a special way into batches
```python
def train_batch(input, target):
    optimizer.zero_grad()
    output = model(input)
    loss = ctc_loss(output, target)
    loss.backward()
    optimizer.step()
```


# Word / Text Line Recognition  
```python
def make_model():
    return nn.Sequential(
        *convolutional_layers(),
        layers.Fun(lambda x: x.sum(2)),
        layers.Conv1d(num_classes, 1)
    )

def train_batch(input, target):
    optimizer.zero_grad()
    output = model(input)
    loss = ctc_loss(output, target)
    loss.backward()
    optimizer.step()     
```


# VGG-Like Model
```python
def make_vgg_model():
    return nn.Sequential(
        layers.Input("BDHW", sizes=[None, 1, None, None]),
        *conv2mp(100, 3, 2, repeat=2),
        *conv2mp(200, 3, 2, repeat=2),
        *conv2mp(300, 3, 2, repeat=2),
        *conv2d(400, 3, repeat=2),
        *project_and_conv1d(800, noutput)
    )
```


# Resnet-Block

- NB: we can easily define Resnet etc. in an object-oriented fashion
```python
def ResnetBlock(d, r=3):
    return nn.Sequential(
        Additive(
            nn.Identity(),
            nn.Sequential(
                nn.Conv2d(d, d, r, padding=r//2), nn.BatchNorm2d(d), nn.ReLU(),
                nn.Conv2d(d, d, r, padding=r//2), nn.BatchNorm2d(d)
            )
        ),
        nn.MaxPool2d(2)
    )

def resnet_blocks(n, d, r=3):
    return [ResnetBlock(d, r) for _ in range(n)]
```


# Resnet-like Model
```python
def make_resnet_model():    
    return nn.Sequential(
        layers.Input("BDHW", sizes=[None, 1, None, None]),
        *conv2mp(64, 3, (2, 1)),
        *resnet_blocks(5, 64), *conv2mp(128, 3, (2, 1)),
        *resnet_blocks(5, 128), *conv2mp(256, 3, 2),
        *resnet_blocks(5, 256), *conv2d(256, 3),
        *project_and_conv1d(800, noutput)
    )
```


# Footprints

- even with projection/1D convolution, a character is first recognized in 2D by the 2D convolutional network
- character recognition with 2D convolutional networks really a kind of deformable template matching
- in order to recognize a character, each pixel at the output of the 2D convolutional network needs to have a footprint large enough to cover the character to be recognized
- footprint calculation:
  - 3x3 convolution, three maxpool operations = 24x24 footprint

FIXME add figure


# Problems with VGG/Resnet+Conv1d

Problem:
- reduces output to H/8, W/8
- CTC alignment needs two pixels for each character
- these models has trouble with narrow characters

Solutions:
- use fractional max pooling
- use upscaling
- use transposed convolutions


# Less Downscaling using `FractionalMaxPool2d`

- permits more max pooling steps without making image too small
- can be performed anisotropically
- necessary non-uniform spacing may have additional benefits
```python
def conv2fmp(d, r, ratio=(0.7, 0.85)):
    return [
        flex.Conv2d(d, r, padding=r//2), flex.BatchNorm2d(), nn.ReLU(),
        nn.FractionalMaxPool2d(3, ratio)
    ]

def make_fmp_model():
    return nn.Sequential(
        layers.Input("BDHW", sizes=[None, 1, None, None]),
        *conv2fmp(50, 3), *conv2fmp(100, 3), *conv2fmp(150, 3), *conv2fmp(200, 3),
        layers.Fun("lambda x: x.sum(2)"), # BDHW -> BDW
        *project_and_conv1d(800, noutput)
    )
```


# Upscaling using `interpolate`

- `interpolate` scales an image, has `backward()`
- `MaxPool2d...interpolate` is a simple multiscale analysis
- can be combined with loss functions at each level
```python
def make_interpolating_model():
    return nn.Sequential(
        layers.Input("BDHW", sizes=[None, 1, None, None]),
        *conv2mp(50, 3), *conv2mp(100, 3), *conv2mp(150, 3), *conv2mp(200, 3),
        layers.Fun("lambda x: x.interpolate(x, scale=4)")
        *project_and_conv1d(800, noutput)
    )
```

<!-- #region -->

# Upscaling using `ConvTranspose1d`


- `ConvTranspose2d` fills in higher resolutions with "templates"
- commonly used in image segmentation and superresolution

<!-- #endregion -->
```python
def make_ct_model():
    return nn.Sequential(
        layers.Input("BDHW", sizes=[None, 1, None, None]),
        *conv2mp(50, 3), 
        *conv2mp(100, 3),
        *conv2mp(150, 3),
        *conv2mp(200, 3),
        layers.Fun("lambda x: x.sum(2)"), # BDHW -> BDW
        flex.ConvTranspose1d(800, 1, stride=2), # <-- undo too tight spacing
        *project_and_conv1d(800, noutput)
    )
```


# How well do these work?

- Works for word or text line recognition.
- All these models only require that characters are arranged left to right.
- Input images can be rotated up to around 30 degrees and scaled.
- Input images can be grayscale.
- Great for scene text and degraded documents.

But:
- You pay a price for translation/rotation/scale robustness: lower performance.

