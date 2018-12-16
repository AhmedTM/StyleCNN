# Style Transfer Net
## Introduction 
   This is an impelementation of Style Transfer Net in the paper [Image Style Transfer Using Convolutional Neural Networks, by Gatys](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf) in PyTorch and from the course [Intro to Deep Learning with PyTorch](https://in.udacity.com/course/deep-learning-pytorch--ud188).
 
 
   In this paper, style transfer uses the features found in the 19-layer VGG Network, which is comprised of a series of convolutional and pooling layers, and a few fully-connected layers.
   
## Instalation
Run this command on your terminal

```
$ git clone https://github.com/AhmedTM/StyleCNN
```
or you can download it directly from github

## Usage
Help:

```
$ python train.py -h
```
Options:

``` 
--content-path        Path to the content image.
--style-path          Path to the style image.
--epochs              Number of iterations to update your image.
--alpha               The content error weight.
--beta                The style error weight.
```
## Images used

Content image              |  Style image              |  Output image
:-------------------------:|:-------------------------:|:-------------------------:
<img src="./img/content.jpg" alt="Content" width="500"/> | <img src="./img/style.jpg" alt="Style" width="500"/> | <img src="./outputs/Ahmed.png" alt="Output" width="500"/>