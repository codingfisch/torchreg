# torchreg

torchreg is a tiny (~300 lines) PyTorch-based library for 2D and 3D image registration.

<p float="left", align="center">
<img src="https://github.com/codingfisch/torchreg/blob/main/examples/alice_big.jpg" width="256"/>
<img src="https://github.com/codingfisch/torchreg/blob/main/examples/alice_small.jpg" width="256"/>
<img src="https://github.com/codingfisch/torchreg/assets/55840648/ddc495fb-f5be-4bba-a8ab-c474a85a34ff" width="256"/>
</p>

## Usage
Affine Registration of two image tensors is done via:
```python
from torchreg import AffineRegistration

# Load images as torch Tensors
big_alice = ...    # Tensor with shape [1, 3 (color channel), 1024 (pixel), 1024 (pixel)]
small_alice = ...  # Tensor with shape [1, 3 (color channel), 1024 (pixel), 1024 (pixel)]
# Intialize AffineRegistration
reg = AffineRegistration(is_3d=False)
# Run it!
moved_alice = reg(big_alice, small_alice)
```

## Features

Multiresolution approach to save compute (per default 1/4 + 1/2 of original resolution for 500 + 100 iterations)
```python
reg = AffineRegistration(scales=(4, 2), iterations=(500, 100))
```
Choosing which operations (translation, rotation, zoom, shear) to optimize
```python
reg = AffineRegistration(with_zoom=False, with_shear=False)
```
Custom initial parameters
```python
reg = AffineRegistration(zoom=torch.Tensor([[1.5, 2.]]))
```
Custom dissimilarity functions and optimizers
```python
def dice_loss(x1, x2):
    dim = [2, 3, 4] if len(x2.shape) == 5 else [2, 3]
    inter = torch.sum(x1 * x2, dim=dim)
    union = torch.sum(x1 + x2, dim=dim)
    return 1 - (2. * inter / union).mean()

reg = AffineRegistration(dissimilarity_function=dice_loss, optimizer=torch.optim.Adam)
```
CUDA support (NVIDIA GPU)
```python
moved_alice = reg(moving=big_alice.cuda(), static=small_alice.cuda())
```

After the registration is run, you can apply it to new images (coregistration)
```python
another_moved_alice = reg.transform(another_alice, shape=(256, 256))
```
with desired output shape.

You can access the affine
```python
affine = reg.get_affine()
```
and the four parameters (translation, rotation, zoom, shear)
```python
translation = reg.parameters[0]
rotation = reg.parameters[1]
zoom = reg.parameters[2]
shear = reg.parameters[3]
```

## Installation
```bash
pip install torchreg
```

## Examples/Tutorials

There are three example notebooks:

- [examples/basics.ipynb](https://github.com/codingfisch/torchreg/blob/main/examples/basic.ipynb) shows the basics by using small cubes/squares as image data
- [examples/images.ipynb](https://github.com/codingfisch/torchreg/blob/main/examples/image.ipynb) shows how to register alice_big.jpg to alice_small.jpg
- [examples/mri.ipynb](https://github.com/codingfisch/torchreg/blob/main/examples/mri.ipynb) shows how to register MR images (Nifti files) including co-, parallel and multimodal registration

## Background

If you want to know how the core of this package works, read [the blog post](https://codingfisch.github.io/2023/08/09/affine-registration-in-12-lines-of-code.html)!

## TODO
- [ ] Add 2D support to SyN, NCC and LinearElasticity
- [ ] Add tests for SyN
