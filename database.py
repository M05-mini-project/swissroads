import tensorflow as tf
import tensorflow_hub as hub
import glob
import os
import PIL.Image as Image
import numpy as np
import pandas as pd

"""
Data set from EPFL Extension School https://www.extensionschool.ch/ 
permission to use given by the EPFL Extension School to Christophe Hoël for the completion of this image recognition exercise
within the course M05 from AI master at Idiap." 
"""

def get_images(folder_name):
    """ 
    Recursively read all folder content, open all images with PIL and create array containing images data.

    Parameters
    ----------
    folder_name : string
        name of the folder containing the images

    Returns:
    ----------
    batches_data : array
        Nx(224x224) array containing resized images in a normalized float32 format
    batches_cat : array
        Nx1 array containing the category of each sample as string
    batches_file : array
        Nx1 array containing the file path of the image sample
    batches_folder : array
        Nx1 array containing the folder type, can either be 'test', 'train' or 'valid'
    """

    folders = ['test','train','valid']
    categories = ['bike', 'car', 'motorcycle','other','truck','van']
    images = []
    for folder in folders:
        for cat in categories:
            images.append(glob.glob('{}/*.png'.format('./' + folder_name + '/' + folder + '/' + cat), recursive=True))

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

    return batches_data, batches_cat, batches_file, batches_folder

def get_batches(X,batch_size):
    """ 
    Split a list into batches of a given size and yield them one by one.

    Parameters
    ----------
    X : list
        List of any kind to be split
    batch_size : int
        The size of each generated batches

    Returns:
    ----------
    batch : list
        a batch of size batch_size, the last batch yielded may not be of this size depending on the lenght of the X input
    """
    for i in range(0, len(X), batch_size):
        yield X[i:i+batch_size]

### helper function to load the mobilnet trained features
def load_data(folder_name):
    """ 
    Get the images present in a given folder, create a database from it and save it in a .npz file.

    A 'output' folder must exist for the database to be saved.

    Parameters
    ----------
    folder_name : string
        name of the folder containing the images
    """

    print('------------------------------------------------')
    print(' LOAD DATABASE.....')
    img_graph = tf.Graph()

    with img_graph.as_default():
        module_url = 'https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/2'
        feature_extractor = hub.Module(module_url)
        input_imgs = tf.placeholder(dtype=tf.float32, shape=[None, 224, 224, 3])
        imgs_features = feature_extractor(input_imgs)
        init_op = tf.group([tf.global_variables_initializer(), tf.tables_initializer()])
        
    img_graph.finalize()

    sess = tf.Session(graph=img_graph)
    sess.run(init_op)

    batches_data, batches_cat, batches_file, batches_folder = get_images(folder_name)
    # Extract features
    features = np.array([])
    i = 0
    for X_batch in get_batches(batches_data, 64):
        images_count = (i+1)*64
        if images_count>len(batches_data):
            images_count = len(batches_data)
        print("Get features from tfhub mobilenet_v2...batch ", i+1, '- images ', images_count, ' out of ',  len(batches_data))
        features_acc = sess.run(imgs_features, feed_dict={input_imgs: X_batch})
        if i == 0:
            features = features_acc            
        else:
            features = np.concatenate((features,features_acc), axis=0 )
        i += 1

    # Append the 3 other features to the 1280 extracted features    
    values = np.append(features, np.array(batches_cat)[:,np.newaxis], axis=1)
    values = np.append(values, np.array(batches_file)[:,np.newaxis], axis=1)
    values = np.append(values, np.array(batches_folder)[:,np.newaxis], axis=1)
    
    # Save the data into a npz file
    columns_name = []
    for i in range(features.shape[1]):
        columns_name.append('feature_' + str(i))
        
    columns_name.append('category')
    columns_name.append('imageFileName')
    columns_name.append('imageSet')

    np.savez('./output/images_data.npz', values=values, columns=columns_name)

    print('------------------------------------------------')
