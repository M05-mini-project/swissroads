import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import PIL.Image as Image
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from scripts import baseline


# to prevent warnings from pandas when slicing dataframe
pd.options.mode.chained_assignment = None


def create_data_sets(df):
    df_train = df[df["imageSet"] == "train"]
    X_tr = df_train.drop(["imageSet", "category", "imageFileName"], axis=1).values

    df_test = df[df["imageSet"] == "test"]
    X_te = df_test.drop(["imageSet", "category", "imageFileName"], axis=1).values

    df_valid = df[df["imageSet"] == "valid"]
    X_va = df_valid.drop(["imageSet", "category", "imageFileName"], axis=1).values

    ### apply cat2num helper function to our data and save into a new column
    df_train.loc[:, "category_num"] = df_train.loc[:, "category"].apply(
        baseline.cat2num
    )
    df_valid.loc[:, "category_num"] = df_valid.loc[:, "category"].apply(
        baseline.cat2num
    )
    df_test.loc[:, "category_num"] = df_test.loc[:, "category"].apply(baseline.cat2num)
    y_tr = df_train.category_num.values
    y_va = df_valid.category_num.values
    y_te = df_test.category_num.values

    return X_tr, y_tr, X_te, y_te, X_va, y_va


def get_batches(X, y, batch_size):
    shuffled_idx = np.arange(len(y))
    np.random.shuffle(shuffled_idx)

    for i in range(0, len(y), batch_size):
        batch_idx = shuffled_idx[i : i + batch_size]
        yield X[batch_idx], y[batch_idx]


def create_execute_TF_graph(epochs, X_tr, X_te, X_va, y_tr, y_te, y_va):
    # ### create a one layer dense network graph using tensorflow:
    # - Create the placeholders for X and y
    # - Create an output layer using tf.layers.dense
    # - Create the Loss function with tf.reduce_mean
    # - Create the optimizer (gradient descend) node with a node for the learning rate
    # - Create a train node to be minimized
    # - Finally compute the predictions
    graph = tf.Graph()
    with graph.as_default():
        # Create placeholders
        X = tf.placeholder(dtype=tf.float32, shape=[None, 1280])
        y = tf.placeholder(dtype=tf.int32, shape=[None])

        # Hidden Layer
        hidden = tf.layers.dense(
            X,
            30,
            activation=tf.nn.relu,
            kernel_initializer=tf.variance_scaling_initializer(scale=2, seed=0),
            bias_initializer=tf.zeros_initializer(),
            name="hidden",
        )

        # Output layer
        logits = tf.layers.dense(
            hidden,
            6,
            activation=None,
            kernel_initializer=tf.variance_scaling_initializer(scale=1, seed=0),
            bias_initializer=tf.zeros_initializer(),
            name="output",
        )

        # Loss function : mean cross-entropy
        mean_ce = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
        )

        # Optimizer : Gradient descent
        lr = tf.placeholder(dtype=tf.float32)
        gd = tf.train.GradientDescentOptimizer(learning_rate=lr)

        # Minimize Loss function : min(cross-entropy)
        train_op = gd.minimize(mean_ce)

        # Compute predictions with the trained model
        predictions = tf.argmax(logits, axis=1, output_type=tf.int32)
        is_correct = tf.equal(y, predictions)
        accuracy = tf.reduce_mean(tf.cast(is_correct, dtype=tf.float32))

        config = tf.ConfigProto(
            inter_op_parallelism_threads=1,
            allow_soft_placement=True,
            device_count={"CPU": 1},
        )

    with tf.Session(graph=graph, config=config) as sess:
        # Initialize variables
        sess.run(tf.global_variables_initializer())

        # Set seed
        np.random.seed(0)

        # Train several epochs
        for epoch in range(epochs):
            # Accuracy values (train) after each batch
            batch_acc = []

            # Get batches of data
            for X_batch, y_batch in get_batches(X_tr, y_tr, 30):
                # Run training and evaluate accuracy
                _, acc_value = sess.run(
                    [train_op, accuracy],
                    feed_dict={X: X_batch, y: y_batch, lr: 0.01},  # Learning rate
                )

                # Save accuracy (current batch)
                batch_acc.append(acc_value)

            # Evaluate validation accuracy
            valid_acc = sess.run(accuracy, feed_dict={X: X_va, y: y_va})

            # Print progress
            print(
                "Epoch {} - valid: {:.3f} train: {:.3f} (mean)".format(
                    epoch + 1, valid_acc, np.mean(batch_acc)
                )
            )

        test_acc = sess.run(accuracy, feed_dict={X: X_te, y: y_te})
        train_acc = sess.run(accuracy, feed_dict={X: X_tr, y: y_tr})

        ## run predictions
        y_te_predict = sess.run(predictions, feed_dict={X: X_te, y: y_te})

    return train_acc, test_acc, y_te_predict


def get_confusion_matrix(y_te, y_values, labels):
    cm = confusion_matrix(y_te, y_values, labels=labels)
    df_res = pd.DataFrame(cm, columns=labels)
    df_res["label"] = labels
    df_res.set_index("label", inplace=True)

    return df_res


# main function to be called by the script results.py to execute the claasification with NN
def main(df, epochs=15):
    ### first create train/test and valid datasets from swissroads data
    X_tr, y_tr, X_te, y_te, X_va, y_va = create_data_sets(df)

    ### scale the data on train dataset
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr.astype(np.float32))
    X_va = scaler.transform(X_va.astype(np.float32))
    X_te = scaler.transform(X_te.astype(np.float32))

    print("---------------------------------------------------")
    print("Neural Network : optimizing ...")

    ### Execute the Graph feeding the NN with the batch-generator function:
    train_acc, test_acc, y_te_predict = create_execute_TF_graph(
        epochs, X_tr, X_te, X_va, y_tr, y_te, y_va
    )
    print("------------------------------------------------")
    print(" Neural network model : ")
    print("train accuray 2-layer dense : {:.3f}".format(train_acc))
    print("test accuray 2-layer dense : {:.3f}".format(test_acc))
    print("------------------------------------------------")

    # confusion matrix
    labels = ["bike", "car", "motorcycle", "other", "truck", "van"]
    df_res = get_confusion_matrix(y_te, y_te_predict, labels)

    print("NN confusion matrix:")
    print(df_res)
    print("------------------------------------------------")

