# neural_style_transfer
Unleash the Power of Neural Style Transfer (NST) to Create Visual Art

# Neural Style Transfer with Tensorflow

## [Example](example)
<p>
  This repository contains an implementation of Neural Style Transfer (NST) using TensorFlow. NST is a deep learning technique that combines the content of one image with the style of another image to create visually appealing artwork. This project serves as a demonstration of my skills in Deep Learning and Generative AI.
</p>
<p align="center">
  <img src="https://assets-global.website-files.com/5d7b77b063a9066d83e1209c/613ebf63d399e5f400cce8f8_neural-style-transfer-basic-structure.png" height="400 width="600" alt="Content Loss">
</p>


## Table of Contents
1. [Introduction](#introduction)
2. [How NST Works](#how-nst-works)
3. [Mathematical Formulas](#mathematical-formulas)
4. [Loss Functions](#loss-functions)
5. [Implementation](#implementation)
6. [Intuition of the Model](#intuition-of-the-model)
7. [Results](#results)
8. [Usage](#usage)
9. [Dependencies](#dependencies)
10. [Acknowledgements](#acknowledgements)

## Introduction

In this project, I've implemented a neural style transfer algorithm using a pretrained VGG-19 network, that takes two input images: a content image and a style image, and produces a third image that combines the content of the content image with the artistic style of the style image. This technique is not only fascinating from an artistic perspective but also showcases the power of deep neural networks.

## How NST Works

NST is based on Convolutional Neural Networks (CNNs) and is achieved through the following steps:
1. Define the content and style representations by extracting feature maps from the CNN.
2. Create a target image and initialize it with the content image.
3. Compute the content loss, which measures how different the content of the target image is from the content image.
4. Compute the style loss, which measures how different the style of the target image is from the style image.
5. Optimize the target image to minimize the content and style losses.

## Mathematical Formulas

### Content Loss
Content loss is defined as the mean squared difference between the feature maps of the content image and the target image at a specific layer of the CNN.
<p align="center">
  <img src="img/content_loss.webp" width="1000" alt="Content Loss">
</p>


### Style Loss
Style loss is computed as the mean squared difference between the Gram matrices of the style image and the target image at multiple layers of the CNN.
<p align="center">
  <img src="img/style_loss.webp" width="1000" alt="Style Loss">
</p>

## Loss Functions

NST optimizes the target image by minimizing a combined loss function that includes both content and style losses.
<p align="center">
  <img src="img/loss_func.webp" width="1000" alt="Total Loss">
</p>

## Implementation

All the images were generated using the Adam optimizer rather than the L-BFGS optimizer used in the original paper due to lack of resources. The content losses were calculated using the conv3_3 layer of the vgg19 network and the style losses were calculated using the conv1_2, conv2_2, conv3_3, conv4_3 layers.
<p align="center">
  <img src="img/pseudo_code.png" width="600" alt="Pseudo Code">
</p>

## Intuition of the Model

The NST model leverages the ability of CNNs to capture high-level content and style features. By minimizing the content and style losses,  our model strives to capture content while embracing style. It's like Picasso collaborating with your vacation photos!
<p align="center">
  <img src="img/gram_matrix_loss_eval.webp" width="1000" alt="GM Evaluation">
</p>

## Results
<div style="display: flex; flex-direction: row; justify-content: center; align-items: center;">
    <img src="images/My-belove.jpg" alt="source image" width="200" style="margin: 10px;" />
    <img src="images/violinonpalette.jpg" alt="style image" height="300" width="200" style="margin: 10px;" />
    <img src="outputs/violinonpalette_onto_My-belove_at_iteration_9.png" alt="NST Output" width="200" style="margin: 10px;" />
</div>

Caption for Image 1 | Caption for Image 2 | Caption for Image 3



The above image showcases the results of applying NST to various content and style image pairs.

## Usage

1. Clone this repository.
2. Install the required dependencies.
3. Run the Jupyter Notebook or Python script to perform NST on your own images.

## Dependencies

- Python 3.x
- TensorFlow
- Keras
- NumPy

## Acknowledgements

This project was inspired by the works of Leon A. Gatys, Alexander S. Ecker, and Matthias Bethge, who introduced the concept of Neural Style Transfer in the paper ["A Neural Algorithm of Artistic Style."](https://arxiv.org/abs/1508.06576)

Feel free to explore the Jupyter Notebook for a step-by-step walkthrough and experiment with different content and style images to generate your own artistic creations.

For a more detailed understanding of the code and the model, please refer to the code files in this repository.

Your contributions and feedback are welcome!
