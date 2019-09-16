```python
%pylab inline
```

```python
import torch
from torch import nn
from torchmore import layers, flex
```


# LAYOUT MODELS



# Layout Analysis

Goals of document layout analysis:

- identify text/image regions on pages
- find text lines
- find words within text lines
- determine reading order



FIXME references and sample images



# RCNN for Layout Analysis

- RCNN has been used for word bounding boxes in scene text
- likely does not work well for whole page segmentation
  - regions often not exactly axis-aligned rectangles
  - regions wildly differ in scale and shape
  - region hypotheses likely difficult to regress from moving windows

FIXME reference



# Layout Analysis as Semantic Segmentation

- semantic image segmentation = image-to-image transformation
- each image is classified into one of several different semantic classes
- possible pixel classes for layout analysis:
  - text / image / table / figure / background
  - baseline / background
  - centerline / background
  - text / image / text line separator / column separator



# What do we label?

Possibilities:

- foreground pixels only
- all pixels in a rectangle / region
- leave it up to the algorithm

Very different results:

- foreground/algorithm = we don't know which pixels belong together
- all pixels in a region = group pixels together by connected components
- labeling / segmentation closely linked to intended post-processing for region extraction



# Page Level Segmentation

Page level segmentation divides images into text regions, image regions, and background.

- precise pixel-level segmentations are not usually needed
- segmentations can be computed at a lower level of resolution
- simple properties like text/image/background can be computed based on local texture / statistics
- separating adjacent text columns or images may be difficult, since background gaps may be narrow

Different uses:

- simply mask out non-text regions for further processing (basic binary map sufficient)
- extract text regions via connected components (requires higher quality segmentation)




# Simple Approach

Word segmentation:

- assume training data consists of page images and word bounding boxes
- create a binary target that is 1 inside word bounding boxes and 0 elsewhere
- learn an image-to-image model predicting the binary map from the input document image

Problem:

- word bounding boxes are often overlapping
- how do we turn the resulting binary image into something we can feed to a word recognizer?



# Centerline / Baseline Approach

Word/line segmentation:

- assume training data consists of page images and word/line bounding boxes
- create a binary image that marks either the center line or the baseline of each bounding box
- learn an image-to-image model predicting the centerline/baseline map from the input document image:

Properties:

- works better than the simple approach
- still need to use non-DL mechanisms to find the word/line bounding boxes from the seeds



# Marker Plus Separator Approach

Word/line segmentation:

- assume training data consists of page images and word/line bounding boxes
- three output classes:
  - background
  - marker (center of bounding box)
  - boundary (outline of bounding box)
- train image-to-image segmentation model to output all three classes
- recover word/line images via marker morphology

Properties:

- functions like RCNN, in that it finds both the location and the size of object instances (words/lines)
- simpler to understand/tune: we can see the marker/boundary proposals



FIXME examples from marker-plus-separator approach



# Footprint and Global Context for Page Segmentation



# Semi-Supervised and Weakly Supervised Approaches

