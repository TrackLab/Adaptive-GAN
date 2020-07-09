# General Information

This General Adversarial Network can quickly be trained on any kind of images. That's why the name "Adaptive".
It has been designed to simply take in the images from the "dataset" folder.
The Project originates from the official Tensorflow Deep Convolutional Generative Adversarial Network Tutorial
https://www.tensorflow.org/tutorials/generative/dcgan

# Usage
Both Grayscale and RGB Images are usable.
Depending on if your dataset is Grayscale or RGB the GAN will train and output as such.
The code includes a simple check wether or not the first Image in the dataset is Grayscale.
Should that check estimate incorrectly, you can overwrite the color mode.

The default image size is 112x112 pixels.
You can change that scale, but then you need to understand how to change the convolutional layers of the generator.

# Important Notes

The code and setup is mostly self explanatory, since most important lines include code for explanation.
If there is something missing or wrong, the code will tell you with simple messages.
During training, the GAN will save its training images after every few epochs. The number of epochs to save a pic after is adjustable.

# Setup

All you need is the python file and a folder called "dataset" containing your images to train on.
The code will resize the images automatically and determine the color space, as mentioned before.
Make sure that the batch size is not more than the amount of images.
The folder "saved_models" and "train_output_images" is created automatically in case it does not exist already.
