# swissroads
This project aims at demonstrating the capacity of machine learning and more precisely Neural Network for image recognition.
It has the particularity to use a advanced features from a trained Neural Network from tensorflow ie 
https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/2

The final scope is to demonstrate the possibility to train a model for image recognition with a few images without starting from scratch but instead using already trained features on large scale dataset train.

The provided dataset is composed of 469 images which is quite few to train a reasonable efficient classifier.
The Mobilnet V2 will be used to get 1280 high level features from our dataset of 469 images.
Using those high level features will allow to perform dissent classification.

# Dataset explanations
The original provided "Swissroads" data set is provided by  - EPFL Extension School.
It is composed of 469 images in format .png file 256x256 pixels.
It contains images of bike,car,motorcycle,truck,van and other categories.
The dataset is already splitted into train/test and Valid sets.

# conda environement creation
A file swissroads_conda_env.yml is provided to first used as follow:</br>
<code>conda env create -f swissroads_conda_env.yml</code></br>
<code>conda activate swissroads_conda_env</code>

# how to start the code
in your newly created environment launch:</br>
<code>python results.py</code>
