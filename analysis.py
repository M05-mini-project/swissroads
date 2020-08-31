# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
import numpy as np
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import PIL.Image as Image

# %% [markdown]
# ### load the npz file and create the array to be then used for model training and testing

# %%
with np.load('images_data.npz', allow_pickle=True) as npz_file:
    df = pd.DataFrame(npz_file['values'], columns= npz_file['columns'])
    
df_train = df[df['imageSet'] == 'train']
X_tr = df_train.drop(['imageSet','category','imageFileName'], axis=1).values
y_tr = df_train.category.values

df_test = df[df['imageSet'] == 'test']
X_te = df_test.drop(['imageSet','category','imageFileName'], axis=1).values
y_te = df_test.category.values

df_valid = df[df['imageSet'] == 'valid']
X_va = df_valid.drop(['imageSet','category','imageFileName'], axis=1).values
y_va = df_valid.category.values

# %% [markdown]
# ### transform the categories into numerical values for NN output

# %%
def cat2num(x):
    if x == 'bike':
        return 0
    if x=='car':
        return 1
    if x =='motorcycle':
        return 2
    if x == 'other':
        return 3
    if x =='truck':
        return 4
    if x =='van':
        return 5
df_train['category_num'] = df_train['category'].apply(cat2num)
df_valid['category_num'] = df_valid['category'].apply(cat2num)
df_test['category_num'] = df_test['category'].apply(cat2num)
y_tr = df_train.category_num.values
y_va = df_valid.category_num.values
y_te = df_test.category_num.values

# %% [markdown]
# ### scale the data on train dataset

# %%
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_tr = scaler.fit_transform(X_tr.astype(np.float32))
X_va = scaler.transform(X_va.astype(np.float32))
X_te = scaler.transform(X_te.astype(np.float32))

# %% [markdown]
# ### create a batch generator function to train our nn.

# %%
def get_batches(X, y, batch_size):
    shuffled_idx = np.arange(len(y))
    np.random.shuffle(shuffled_idx)
    
    for i in range(0, len(y), batch_size):
        batch_idx = shuffled_idx[i:i+batch_size]
        yield X[batch_idx], y[batch_idx]

# %% [markdown]
# ### create a one layer dense network graph using tensorflow:
# - Create the placeholders for X and y
# - Create an output layer using tf.layers.dense
# - Create the Loss function with tf.reduce_mean
# - Create the optimizer (gradient descend) node with a node for the learning rate
# - Create a train node to be minimized
# - Finally compute the predictions

# %%
import tensorflow as tf

graph1 = tf.Graph()

with graph1.as_default():
    # Create placeholders
    X = tf.placeholder(dtype=tf.float32, shape=[None, 1280])
    y = tf.placeholder(dtype=tf.int32, shape=[None])

    # Output layer
    logits = tf.layers.dense(
        X, 6, activation=None, #6 categories as output
        kernel_initializer=tf.variance_scaling_initializer(scale=1, seed=0),
        bias_initializer=tf.zeros_initializer(),
        name='output'
    )
    
    #Loss function : mean cross-entropy
    mean_ce = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits))
    
    #Optimizer : Gradient descent
    lr = tf.placeholder(dtype=tf.float32)
    gd = tf.train.GradientDescentOptimizer(learning_rate = lr)
    
    #Minimize Loss function : min(cross-entropy)
    train_op = gd.minimize(mean_ce)
    
    #Compute predictions with the trained model
    predictions = tf.argmax(logits, axis=1, output_type=tf.int32)
    is_correct = tf.equal(y, predictions)
    accuracy = tf.reduce_mean(tf.cast(is_correct, dtype=tf.float32))

# %% [markdown]
# ### Execute the Graph feeding the NN with the batch-generator function:
# - train the model running our train_op node at each epoch
# - calculate the accuracy with valid dataset at each epoch
# -  calculate the accuracy of the final NN with test dataset

# %%
valid_acc_values=[]

with tf.Session(graph=graph1) as sess:
    # Initialize variables
    sess.run(tf.global_variables_initializer())

    # Set seed
    np.random.seed(0)

    # Train several epochs
    for epoch in range(15):
        # Accuracy values (train) after each batch
        batch_acc = []

        # Get batches of data
        for X_batch, y_batch in get_batches(X_tr, y_tr, 30):
            # Run training and evaluate accuracy
            _, acc_value = sess.run([train_op, accuracy], feed_dict={
                X: X_batch,
                y: y_batch,
                lr: 0.01 # Learning rate
            })

            # Save accuracy (current batch)
            batch_acc.append(acc_value)

        # Evaluate validation accuracy
        valid_acc = sess.run(accuracy, feed_dict={
            X: X_va,
            y: y_va
        })
        valid_acc_values.append(valid_acc)
        
        # Print progress
        print('Epoch {} - valid: {:.3f} train: {:.3f} (mean)'.format(
            epoch+1, valid_acc, np.mean(batch_acc)
        ))
        
    test_acc = sess.run(accuracy, feed_dict={
            X: X_te,
            y: y_te
        })
    print('test accuray 1-layer dense : {:.3f}'.format(test_acc))

# %% [markdown]
# - the test accuracy for a 1-layer dense NN is 0.9
# %% [markdown]
# ### create a 2-layer Dense NN :
# - add a hidden layer to the previous nn graph

# %%
import tensorflow as tf

graph2 = tf.Graph()

with graph2.as_default():
    # Create placeholders
    X = tf.placeholder(dtype=tf.float32, shape=[None, 1280])
    y = tf.placeholder(dtype=tf.int32, shape=[None])
    
    # Hidden Layer
    hidden = tf.layers.dense(
        X,30,activation=tf.nn.relu,
        kernel_initializer=tf.variance_scaling_initializer(scale=2, seed=0),
        bias_initializer=tf.zeros_initializer(),
        name='hidden'
    ) 

    # Output layer
    logits = tf.layers.dense(
        hidden, 6, activation=None,
        kernel_initializer=tf.variance_scaling_initializer(scale=1, seed=0),
        bias_initializer=tf.zeros_initializer(),
        name='output'
    )
    
    #Loss function : mean cross-entropy
    mean_ce = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits))
    
    #Optimizer : Gradient descent
    lr = tf.placeholder(dtype=tf.float32)
    gd = tf.train.GradientDescentOptimizer(learning_rate = lr)
    
    #Minimize Loss function : min(cross-entropy)
    train_op = gd.minimize(mean_ce)
    
    #Compute predictions with the trained model
    predictions = tf.argmax(logits, axis=1, output_type=tf.int32)
    is_correct = tf.equal(y, predictions)
    accuracy = tf.reduce_mean(tf.cast(is_correct, dtype=tf.float32))


# %%
valid_acc_values=[]

with tf.Session(graph=graph2) as sess:
    # Initialize variables
    sess.run(tf.global_variables_initializer())

    # Set seed
    np.random.seed(0)

    # Train several epochs
    for epoch in range(15):
        # Accuracy values (train) after each batch
        batch_acc = []

        # Get batches of data
        for X_batch, y_batch in get_batches(X_tr, y_tr, 30):
            # Run training and evaluate accuracy
            _, acc_value = sess.run([train_op, accuracy], feed_dict={
                X: X_batch,
                y: y_batch,
                lr: 0.01 # Learning rate
            })

            # Save accuracy (current batch)
            batch_acc.append(acc_value)

        # Evaluate validation accuracy
        valid_acc = sess.run(accuracy, feed_dict={
            X: X_va,
            y: y_va
        })
        valid_acc_values.append(valid_acc)
        
        # Print progress
        print('Epoch {} - valid: {:.3f} train: {:.3f} (mean)'.format(
            epoch+1, valid_acc, np.mean(batch_acc)
        ))
        
    test_acc = sess.run(accuracy, feed_dict={
            X: X_te,
            y: y_te
        })
    print('test accuray 2-layer dense : {:.3f}'.format(test_acc))

# %% [markdown]
# - the test accuracy for a 2-layers dense NN is 0.96

# %%



