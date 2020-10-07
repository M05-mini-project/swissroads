.. image:: https://travis-ci.org/M05-mini-project/swissroads.svg?branch=master
   :target: https://travis-ci.org/M05-mini-project/swissroads

.. image:: https://coveralls.io/repos/github/M05-mini-project/swissroads/badge1.svg
   :target: https://coveralls.io/github/M05-mini-project/swissroads?branch=master

.. image:: https://img.shields.io/badge/docs-latest-orange.svg
   :target: https://m05-mini-project.github.io/swissroads/

.. image:: https://img.shields.io/badge/github-project-0000c0.svg
   :target: https://https://github.com/M05-mini-project/swissroads/

.. image:: https://img.shields.io/badge/pypi-project-blueviolet.svg
   :target: https://pypi.org/project/rr_swissroads

=======================================================================
 **Transfer learning : Use of advanced feature for image recognition**
=======================================================================


1- Swissroads project (Transfer learning)
-----------------------------------------

This project aims at demonstrating the capacity of machine learning and more precisely Neural Network for image recognition.
It has the particularity to use a advanced features from a trained Neural Network from tensorflow ie 
https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/4

The final scope is to demonstrate the possibility to train a model for image recognition with a few images without starting from scratch but instead using already trained features on large scale dataset train.

The provided dataset is composed of 469 images which is quite few to train a reasonable efficient classifier.
The Mobilnet V2 will be used to get 1280 high level features from our dataset of 469 images.
Using those high level features will allow to perform dissent classification.

2- Dataset explanations
-------------------------

The original provided "Swissroads" data set is provided by  - EPFL Extension School.
It is composed of 469 images in format .png file 256x256 pixels.
It contains images of bike,car,motorcycle,truck,van and other categories.
The dataset is already splitted into train/test and Valid sets.

3- Conda environement creation
-------------------------------

A file swissroads_conda_env.yml is provided to first used as follow::

    conda env create -f swissroads_conda_env.yml
    conda activate swissroads_conda_env

4- How to start the program?
-----------------------------

in your newly created environment launch::

    python results.py
