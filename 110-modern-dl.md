
# MODERN DEEP LEARNING ERA


# Breakthrough Paper

"ImageNet Classification with Deep Convolutional Neural Networks", Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton, 2013

- substantially better performance on image recognition than non-neural approaches
- effective use of GPUs for convolutional neural networks
- ReLU nonlinearity
- max pooling
- data augmentation
- dropout, local response normalization
- much bigger than previous networks

None of these were new ideas, but this paper brought it all together.


# Alexnet Architecture

![alexnet](figs/alexnet.png)


# Alexnet Architecture (approximately)

    nn.Sequential(
        layers.Input("BDHW", size=(None, 3, 224, 224)),
        flex.Conv2d(2*48, (11, 11)),
        nn.ReLU(),
        nn.MaxPool2d((2, 2)),
        flex.Conv2d(2*192, (3, 3)),
        nn.ReLU(),
        flex.Conv2d(2*192, (3, 3)),
        nn.ReLU(),
        flex.Conv2d(2*128, (3, 3)),
        layers.Reshape(0, [1, 2, 3]),
        flex.Linear(4096),
        nn.ReLU(),
        flex.Linear(4096),
        nn.ReLU(),
        flex.Linear(1000)
    )



# Alexnet Learned Features

![alexnet features](figs/alexnet-features.png)



# ReLU

![relu](figs/relu-nonlinearity.png)

$\sigma(x) = (1 + e^{-x})^{-1}$, $\rho(x) = \max(0, x)$


# ReLU Derivatives

![relu deriv](figs/relu-deriv.png)


# Nonlinearity Properties

| property          | sigmoid          | ReLU              |
|-------------------|------------------|-------------------|
| derivatives       | infinite         | f': discontinuous |
|                   |                  | f'': zero         |
| monotonicity      | monotonic        | monotonic         |
| range             | $(0, 1)$         | $(0, \infty)$     |
| zero-derivative   | none             | $(-\infty, 0)$    |


# ReLU

- much faster to compute
- converges faster
- scale independent
- existence of zero-derivative regions causes "no deltas", units may "go dead"
- positive output only
- results in piecewise linear approximations to functions
- results in classifiers based on linear arrangements


# Max Pooling

![max pooling](figs/maxpool.png)

- replaces average pooling, reduces resolution
- performed per channel
- nonlinear operation, somewhat similar to morphological operations


# Local Response Normalization

$$ y = x \cdot (k + \alpha (K * |x|^\gamma) ^ \beta)^-1 $$

- Here, $*$ is a convolution operation.
- That is, we normalize the image with an average of the local response. 
- In Alexnet, $k=2$, K is a 5x5 pillbox, $\gamma=2$, $\beta=0.75$
- A simple variance normalization would use $k=0$, $\gamma=2$, and $\beta=0.5$

In later models, this is effectively replaced by batch normalization.


# Dropout

- randomly turn off units during training
- motivated by an approximation to an ensemble of networks
- intended to lead to better generalization from limited samples


# Data Augmentation

- generate additional training samples by modifying the original images
- long history in OCR
- for image training:
  - random geometric transformation
  - random distortions
  - random photometric transforms
  - addition of noise, distractors, masking


# FURTHER DEVELOPMENTS


# Batch Normalization

"Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift", S. Ioffe and C. Szegedy, 2015.

- attributes slower learning to "internal covariate shift"
- suggests that ideally, each layer should "whiten" the data
- instead normalizes mean and variance for each neuron
- normalization based on batch statistics (and running statistics)


# Inception

Szegedy, Christian, et al. "Rethinking the inception architecture for computer vision." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.

- very deep architecture built up from complex modules
- separable convolutions for large footprints
- "label smoothing"


# VGG Networks

Simonyan, Karen, and Andrew Zisserman. "Very deep convolutional networks for large-scale image recognition." arXiv preprint arXiv:1409.1556 (2014).

- very deep networks with fairly regular structure
- multiple convolutions + max pooling
- combined with batch normalization in later systems


# Residual Networks

He, Kaiming, et al. "Deep residual learning for image recognition." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.

- novel architecture composed of modules
- each module consists of convolutional layers
- the output of the convolutional layers is added to the input
- ReLU and batch normalization is used throughout


# LOCALIZATION


# Localization of Objects

- objects occur at different locations in scenes/images
- different strategies with recognizing objects:
  - global classification
  - moving/scanning window
  - region proposals (RCNN etc.)
  - learning dense markers / segmentation


# Global Classification
```python
def conv2d_block(d):
    return nn.Sequential(
        flex.Conv2d(d, 3, padding=1), flex.BatchNorm2d(), flex.ReLU(),
        flex.Conv2d(d, 3, padding=1), flex.BatchNorm2d(), flex.ReLU(),
        flex.MaxPool2d(),
    )

def make_model():
    return nn.Sequential(
        *conv2d_block(64), *conv2d_block(128), *conv2d_block(256), 
        *conv2d_block(512), *conv2d_block(1024), *conv2d_block(2048),
        # we have a (None, 2048, 4, 4) batch at this point
        layers.Reshape(0, [1, 2, 3]),
        flex.Linear(4096), flex.BatchNorm(), nn.ReLU(),
        flex.Linear(4096), flex.BatchNorm(), nn.ReLU(),
        flex.Linear(1000)
    )
```


# Sliding Windows

![overfeat](figs/overfeat.png)

Sermanet, Pierre, et al. "Overfeat: Integrated recognition, localization and detection using convolutional networks." arXiv preprint arXiv:1312.6229 (2013).




# Region Proposal Network

![region proposal](figs/region-proposal.png)

Ren, Shaoqing, et al. "Faster r-cnn: Towards real-time object detection with region proposal networks." Advances in neural information processing systems. 2015.

