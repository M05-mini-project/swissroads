import tensorflow as tf
import tensorflow_hub as hub
import glob
import os
import PIL.Image as Image
import numpy as np
import pandas as pd


""" It is the first step to execute when reproducing the project.
    The database.py script is ised to get the data from swissroads and the high level features from 
    https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/2
    once retrieved, all the infornmation will be saved into a .npz file called images_data.npz """

### Helper function to laod the provided image datsaet in "swissroads folder"
def get_images():
    """ - Let's loop all the dataset folders *.png files to create features
        - parse all files into swissroads folder
        - open the image with PIL and resize to the desired size for Mobilnet V2 ie 224x224, normalized between 0 and 1 """

    folders = ['test','train','valid']
    categories = ['bike', 'car', 'motorcycle','other','truck','van']
    images = []
    for folder in folders:
        for cat in categories:
            images.append(glob.glob('{}/*.png'.format('./swissroads_images/' + folder + '/' + cat), recursive=True))

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

# Function to get batches of data
def get_batches(X,batch_size):
    for i in range(0, len(X), batch_size):
        yield X[i:i+batch_size]

### helper function to load the mobilnet trained features
def load_data():

    """ The Helper function load_data performs 2 major operations:
    - call get_images which import the image files
    - Create a TensorFlow graph to extract the features from Mobilenet V2 """

    print('------------------------------------------------')
    print(' LOAD DATABASE.....')
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

    # Run the Tensor Flow Graph to textract the features
    # Create a TF session
    sess = tf.Session(graph=img_graph)

    # Initialize it
    sess.run(init_op)

    #get images 
    batches_data, batches_cat, batches_file, batches_folder = get_images()

    # Extract features
    features = np.array([])
    i = 0
    for X_batch in get_batches(batches_data, 64):
        print("Get features from tfhub mobilenet_v2...batch ", i+1, '- images ', (i+1)*64, ' out of ',  len(batches_data))
        features_acc = sess.run(imgs_features, feed_dict={input_imgs: X_batch})
        if i == 0:
            features = features_acc            
        else:
            features = np.concatenate((features,features_acc), axis=0 )
        i += 1

    # Append the 3 other features previously explained to the 1280 extracted features    
    values = np.append(features, np.array(batches_cat)[:,np.newaxis], axis=1)
    values = np.append(values, np.array(batches_file)[:,np.newaxis], axis=1)
    values = np.append(values, np.array(batches_folder)[:,np.newaxis], axis=1)

    # - We finally have a 469 images with 1280 high level features extracted from Mobilnet v2 each
    # - added 3 features to recover easily the category of the image, the file source and the dataset
    
    # Finally Save the data into a npz file
    columns_name = []
    for i in range(features.shape[1]):
        columns_name.append('feature_' + str(i))
        
    columns_name.append('category')
    columns_name.append('imageFileName')
    columns_name.append('imageSet')

    np.savez('./output/images_data.npz', values=values, columns=columns_name)

    print('------------------------------------------------')
