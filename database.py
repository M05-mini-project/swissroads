# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
import tensorflow as tf
import tensorflow_hub as hub
import glob
import os
import PIL.Image as Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# # 1- Let's loop all the dataset folders *.png files to create features
#     - my idea is to create a unique npz file containing the 1280 features extrated from Mobilnet V2
#     - and add 3 columns with :
#         - image category (bike, car,...)
#         - image file name (xxx.png)
#         - image set (train, valid, test)

# - parse all files into swissroads folder
# - open the image with PIL and resize to the desired size for Mobilnet V2 ie 224x224, normalized between 0 and 1


folders = ['test','train','valid']
categories = ['bike', 'car', 'motorcycle','other','truck','van']
images = []
for folder in folders:
    for cat in categories:
        images.append(glob.glob('{}/*.png'.format('./swissroads/' + folder + '/' + cat), recursive=True))

batches_data = []
batches_cat = []
batches_file = []
batches_folder = []

for j in range(len(folders)):
    for i in range(len(categories)):
        for image in images[i+6*j]:
            with open(image,"rb") as file:
                img = Image.open(file)
                img_resized = img.resize([224, 224], resample=Image.BILINEAR)       
                img_batch = np.array(img_resized, dtype=np.float32)[:, :, :]/255
                batches_data.append(img_batch)
                batches_cat.append(categories[i])
                batches_file.append(file.name) 
                batches_folder.append(folders[j])

# # 2- Create a TensorFlow graph to extract the features from Mobilenet V2
# Create a TF graph
img_graph = tf.Graph()

with img_graph.as_default():
    # Download the mobilnet_v2 module
    module_url = 'https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/2'
    feature_extractor = hub.Module(module_url)

    # Create input placeholder
    input_imgs = tf.placeholder(dtype=tf.float32, shape=[None, 224, 224, 3])

    # Create a node with the features of the input image
    imgs_features = feature_extractor(input_imgs)

    # Collect initializers
    init_op = tf.group([tf.global_variables_initializer(), tf.tables_initializer()])
    
img_graph.finalize()

# %% [markdown]
# # 3- Run the Tensor Flow Graph to textract the features

# %%
# Create a TF session
sess = tf.Session(graph=img_graph)

# Initialize it
sess.run(init_op)

# Extract features
features = sess.run(imgs_features, feed_dict={input_imgs: batches_data})
features.shape

# %% [markdown]
# # 4- Append the 3 other features previously explained to the 1280 extracted features

# %%
values = np.append(features, np.array(batches_cat)[:,np.newaxis], axis=1)
values = np.append(values, np.array(batches_file)[:,np.newaxis], axis=1)
values = np.append(values, np.array(batches_folder)[:,np.newaxis], axis=1)


# %%
values.shape

# %% [markdown]
# - We finally have a 469 images with 1280 high level features extracted from Mobilnet v2 each
# - added 3 features to recover easily the category of the image, the file source and the dataset
# %% [markdown]
# # 5-save the data into a npz file

# %%
columns_name = []
for i in range(1280):
    columns_name.append('feature_' + str(i))
    
columns_name.append('category')
columns_name.append('imageFileName')
columns_name.append('imageSet')

np.savez('images_data.npz', values=values, columns=columns_name)


# %%



