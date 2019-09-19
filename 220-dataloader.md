---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.1'
      jupytext_version: 1.2.3
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

```python slideshow={"slide_type": "skip"}
%pylab inline
rc("image", cmap="gray", interpolation="bicubic")
```

```python slideshow={"slide_type": "skip"}
figsize(8,8)
```

<!-- #region {"slideshow": {"slide_type": "slide"}} -->
# INPUT PIPELINES
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "slide"}} -->
# PyTorch

Classes:

- `torch.utils.data.DataSet` -- random access data sets
- `torch.utils.data.IterableDataset` -- sequential access data sets
- `torch.utils.data.DataLoader` -- multithreaded loading, augmentation, batching

Storage:

- usually as individual files or in LMDB
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "slide"}} -->
# TensorFlow

- `TFRecord` / `tf.Example` -- sequential records of serialized data
- used with sharding and object store for large datasets inside Google
- not a lot of tools available outside Google
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "slide"}} -->
# PYTORCH DATA LOADING
<!-- #endregion -->

```python slideshow={"slide_type": "slide"}
from torch.utils.data import Dataset, IterableDataset, DataLoader
from torchvision import datasets, transforms

mnist = datasets.MNIST(root="./__CACHE__", download=True)

print(mnist[0])
```

<!-- #region {"slideshow": {"slide_type": "fragment"}} -->
- Data in PyTorch is accessed through the `Dataset` or `IterableDataset` classes.
- A `Dataset` behaves just like an array, although in practice, it often loads data from disk.
<!-- #endregion -->

```python slideshow={"slide_type": "slide"}
image, label = mnist[0]
figsize(4, 4); imshow(image)
```

<!-- #region {"slideshow": {"slide_type": "fragment"}} -->
- `Dataset` items are usually pairs of input and target
- Here, we have an input image in `PIL` format and an integer class label
<!-- #endregion -->

```python slideshow={"slide_type": "slide"}
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (1.0,))
])

mnist = datasets.MNIST(transform=transform, train=True, root="./__CACHE__", download=True)
image, label = mnist[0]
print(type(image), image.shape, image.dtype, image.min().item(), image.max().item())
imshow(image[0])
```

<!-- #region {"slideshow": {"slide_type": "fragment"}} -->
- We can specify transformations to be inserted after the initial dataset loading.
- These transformations of perform augmentation and conversion to Torch tensors.
<!-- #endregion -->

```python slideshow={"slide_type": "slide"}
loader = DataLoader(mnist, batch_size=16, shuffle=True)
images, labels = next(iter(loader))
print(type(images), images.shape, images.dtype, images.min().item(), images.max().item())
```

<!-- #region {"slideshow": {"slide_type": "fragment"}} -->
- PyTorch models are usually trained on batches.
- The `Dataloader` class takes care of batching.
<!-- #endregion -->

```python slideshow={"slide_type": "slide"}
loader = DataLoader(mnist, batch_size=16, shuffle=True, num_workers=8)
images, labels = next(iter(loader))
print(type(images), images.shape, images.dtype, images.min().item(), images.max().item())
```

<!-- #region {"slideshow": {"slide_type": "fragment"}} -->
- Data loading, decompression, and augmentation is often compute intensive and becomes a bottleneck.
- To speed up data loading, the `DataLoader` class can use multiple workers.
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "slide"}} -->
# NVIDIA Dali & Tensorcom

- With `DataLoader` all workers still run on the same machine, and the CPU may become a bottleneck.
- NVIDIA Dali offloads a lot of data augmentation to the GPU and can speed up preprocessing substantially.
- Tensorcom offloads data augmentation to other hosts, allowing a large number of CPUs to feed a single GPU.
- Tensorcom also permits the use of RDMA.
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "slide"}} -->
# PyTorch Dataset / DataLoader

- widely used, works well on SSD
- `Dataset` is random access = lots of small reads and seeks for Imagenet
- poor performance on network file system and rotational disks
- image decompression and augmentation very CPU intensive
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "fragment"}} -->
# WebDataset
<!-- #endregion -->

```python
!unset GZIP; curl -s http://storage.googleapis.com/lpr-imagenet/imagenet_train-0000.tgz | tar -ztvf - | sed 5q
```

- `WebDataset` stores data not as individual files, but as standard POSIX tar archives.
- File formats are otherwise unchanged.

```python slideshow={"slide_type": "slide"}
from webdataset import WebDataset
imagenet = WebDataset("http://storage.googleapis.com/lpr-imagenet/imagenet_train-0000.tgz",
                      extensions="jpg cls")
```

<!-- #region {"slideshow": {"slide_type": "fragment"}} -->
- Reading a `WebDataset` is as simple as specifying a URL and the extensions you want extract.
- Data can be stored locally (`file:`), on web servers (`http:`, `https:`), and in cloud storage (`gs:`, `s3:`)
<!-- #endregion -->

```python slideshow={"slide_type": "slide"}
for image, cls in imagenet:
    imshow(image)
    print(cls)
    break
```

<!-- #region {"slideshow": {"slide_type": "fragment"}} -->
- `WebDataset` allows you to iterate through your samples just like a regular PyTorch dataset
- It has convenient file-based rules for decoding built-in.
<!-- #endregion -->

```python slideshow={"slide_type": "skip"}
print(WebDataset.__base__)
imagenet[0]
```

<!-- #region {"slideshow": {"slide_type": "skip"}} -->
`WebDataset` is derived from `IterableDataset`, a new dataset type in PyTorch 1.2. It cannot be indexed, only iterated over.
<!-- #endregion -->

```python slideshow={"slide_type": "slide"}
from webdataset import WebDataset
imagenet = WebDataset("http://storage.googleapis.com/lpr-imagenet/imagenet_train-{0000..0147}.tgz",
                      extensions="jpg cls", decoder="torch")

for image, cls in imagenet:
    break
imshow(image.permute(1, 2, 0))
print(cls)
print(image.shape, image.min().item(), image.max().item(), image.dtype)
```

- You can decode automatically to NumPy (default), Torch, and PIL.
- You can decode images to 8 bit or float, grayscale or color
- If you want to handle decoding completely yourself, just set `decoder=None`

```python slideshow={"slide_type": "slide"}
from torchvision import transforms

augment = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])
```

<!-- #region {"slideshow": {"slide_type": "fragment"}} -->
- The `torchvision` package contains a number of common data transformations and augmentations.
- This pipeline is commonly used with Imagenet training.
- These generally operate on PIL image types.
<!-- #endregion -->

```python slideshow={"slide_type": "slide"}
from webdataset import WebDataset
imagenet = WebDataset("http://storage.googleapis.com/lpr-imagenet/imagenet_train-{0000..0147}.tgz",
                      extensions="jpg cls",
                      decoder="pil",
                      transforms=[augment, lambda x: x-1])

for image, cls in imagenet:
    break
imshow(image.permute(1, 2, 0))
print(cls)
```

<!-- #region {"slideshow": {"slide_type": "fragment"}} -->
- Insert the transforms using the `transforms=` argument.
- Note that unlike `Dataset`, `WebDataset` takes a list of transforms corresponding to each element of the output tuple.
<!-- #endregion -->

```python slideshow={"slide_type": "slide"}
from webdataset import WebDataset
imagenet = WebDataset("http://storage.googleapis.com/lpr-imagenet/imagenet_train-{0000..0147}.tgz",
                      shuffle=100,
                      extensions="jpg cls",
                      decoder="pil",
                      transforms=[augment, lambda x: x-1])

for image, cls in imagenet:
    break
imshow(image.permute(1, 2, 0))
print(cls)
```

<!-- #region {"slideshow": {"slide_type": "fragment"}} -->
- Web datasets are usually split up into multiple shards.
- Shards are referred to using standard Bash brace expansion syntax.
- This brace expansion is carried out internally.
- Sharding is important both for achiving faster I/O speeds and for shuffling data.
<!-- #endregion -->

```python slideshow={"slide_type": "slide"}
imagenet = WebDataset("http://storage.googleapis.com/lpr-imagenet/imagenet_train-{0000..0147}.tgz",
                      extensions="jpg cls",
                      decoder="pil",
                      transforms=[augment, lambda x: x-1])
loader = DataLoader(imagenet, batch_size=16, num_workers=8)
images, classes = next(iter(loader))
print(images.size(), classes.size())
```

<!-- #region {"slideshow": {"slide_type": "fragment"}} -->
- `WebDataset` combines with `DataLoader` and parallel augmentation just like any other `Dataset`
- When using multiple workers, each worker is assigned a distinct shard.
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "slide"}} -->
# WebDataset

- just store datasets as tar files on web server, object store, etc.
- drop-in replacement for PyTorch's `Dataset`
- same preprocessing methods

Advantages:
- sequentials reads -- much faster on rotational drives
- achieves SSD-like performance for petabyte datasets on network shares
- easier to manage than millions of files
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "slide"}} -->
# TARPROC
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "slide"}} -->
The `tarproc` package contains a number of tools that operate on datasets stored as tar files:

- tar2tsv -- extract data from tar files and put into CSV format
- tarcat -- concatenate tar files
- tarcreate -- create a tar file from a CSV/TSV plan
- tarfirst -- extract the first file matching some criteria
- tarpcat -- parallel tar concatenation/shuffle
- tarproc -- xargs running over tar files
- tarshow -- display contents of tar files as text or images
- tarsort -- sort the contents of a tar file by key
- tarsplit -- split a tar file into shards
<!-- #endregion -->

```python slideshow={"slide_type": "slide"}
!gsutil cat gs://lpr-imagenet/imagenet_train-0000.tgz | tarscat -c 10 > small.tar

!tar tvf small.tar | fgrep .jpg | wc -l
```

<!-- #region {"slideshow": {"slide_type": "fragment"}} -->
Tar files can be pipelined, just like other kinds of UNIX jobs.

`tarscat -c 10` reads 10 samples from the input, writes them to the output, and quits.
<!-- #endregion -->

```python slideshow={"slide_type": "slide"}
!tar2tsv -f cls small.tar
```

<!-- #region {"slideshow": {"slide_type": "fragment"}} -->
`tar2tsv` extracts data from tar files and tabulates it.
<!-- #endregion -->

```python slideshow={"slide_type": "slide"}
!tar -cf - --sort=name -C /mdata/imagenet-raw train | tarsplit -s 1e9 -o imagenet_train --maxshards 2
```

<!-- #region {"slideshow": {"slide_type": "fragment"}} -->
To create sharded tar files, tar up your original dataset (either with `tar --sorted` or `tarpcat`) and pipe it to `tarsplit`.
<!-- #endregion -->

```python slideshow={"slide_type": "slide"}
!tarshow  -c 2 small.tar
```

<!-- #region {"slideshow": {"slide_type": "fragment"}} -->
`tarshow` will show the contents of a tar file, and optionally display images.
<!-- #endregion -->

```python slideshow={"slide_type": "slide"}
!tarproc -p 5 -c 'ls; gm mogrify -size 256x256 *.jpg -resize 256x256' small.tar > out.tar
```

<!-- #region {"slideshow": {"slide_type": "fragment"}} -->
`tarproc` allows you to map shell scripts of samples comprising a dataset. Here, we resize all the `.jpg` files in our dataset to a given size using the GraphicsMagick `mogrify` tool.

This code also runs in parallel on 5 cores.
<!-- #endregion -->

```python slideshow={"slide_type": "slide"}
#kujob -s '{0000..0999}' -- tarproc -p 16 -c 'run-ocr *.png' gs://g1000/scanned-{}.tar -o gs://g1000/ocr-{}.tar
```

<!-- #region {"slideshow": {"slide_type": "fragment"}} -->
Combined with a job queueing system (here Kubernetes `kujob`), we can easily process very large sharded datasets.

This runs an OCR engine over 1000 shards representing the 1000 scanned books in the Google 1000 Books dataset; each shard is run in parallel as a separate job, and within each job, 16 pages are processed in parallel.
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "slide"}} -->
# PYTHON BASED MAP-REDUCE
<!-- #endregion -->

```python slideshow={"slide_type": "slide"}
from webdataset.writer import TarWriter
sink = TarWriter("out2.tar")

def process_sample(sample):
    return dict(__key__=sample["__key__"],
                  png=sample["jpg"].resize((256, 256)),
                  cls=str(sample["cls"]))

for sample in WebDataset("small.tar", decoder="pil"):
    sink.write(process_sample(sample))

sink.close()
```

<!-- #region {"slideshow": {"slide_type": "fragment"}} -->
Processing many files is also easy from Python. Here, we resize the `.jpg` file in each sample using PIL, then write teh output to another tar file. This is all fast, sequential dataset reading.
<!-- #endregion -->

```python slideshow={"slide_type": "slide"}
import multiprocessing as mp
pool = mp.Pool(8)

processed = pool.imap_unordered(process_sample, WebDataset("small.tar", decoder="pil"))

with TarWriter("out2.tar") as sink:
    for sample in processed: sink.write(sample)
```

<!-- #region {"slideshow": {"slide_type": "fragment"}} -->
Combined with Python `multiprocessing`, you can perform large scale parallel processing of shards.
<!-- #endregion -->

```python slideshow={"slide_type": "skip"}
def distributed_map_unordered(*args, **kw): pass
```

```python slideshow={"slide_type": "slide"}
import multiprocessing as mp
pool = mp.Pool(8)

def process_shard(input_shard, output_shard):
    processed = pool.imap_unordered(process_sample, WebDataset(input_shard, decoder="pil"))
    with TarWriter(output_shard) as sink:
        for sample in processed: sink.write(sample)

shards = [(f"gs://mybucket/original-{i:04d}.tar", f"gs://mybucket/rescaled-{i:04d}.tar")
              for i in range(1000)]
            
distributed_map_unordered(process_shard, shards)
```

<!-- #region {"slideshow": {"slide_type": "fragment"}} -->
You can also use your favorite distributed queueing and processing framework for Python to execute very large scale data processing jobs just from Python.
<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "slide"}} -->
# Large Scale, Multi-Server Training

<img src="figs/distributed.jpg" width=800>

<!-- #endregion -->

<!-- #region {"slideshow": {"slide_type": "slide"}} -->
# Large Scale Distributed Training

- commonly 16-256 GPUs training from the same dataset
- 200-1000 GB/s input data rate per GPU
- disk : GPU ratio of 2:1 to 10:1
- CPU node : GPU node ratio of 2:1
- on average, each disk only used by at most one GPU
- data is cached from archival storage to dedicated disks
- all protocols are HTTP (WebDav for management operations)
<!-- #endregion -->
