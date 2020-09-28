import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import PIL.Image as Image
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def plot_images(df_test, categories):
    # Create a grid of subplots
    fig, axes = plt.subplots(nrows=6, ncols=3)

    # Remove gaps between suplots
    plt.subplots_adjust(wspace=0, hspace=0)

    for i in range(len(categories)):
        df2 = df_test[df_test["category"] == categories[i]]
        for j in range(3):
            axis = axes.flatten()[3 * i + j]
            img = Image.open(df2["imageFileName"].iloc[j])

            axis.imshow(img)
            axis.get_xaxis().set_visible(False)
            axis.get_yaxis().set_visible(False)

    plt.savefig("./output/images_display.png")


def apply_PCA(df, categories):

    df_train = df[df["imageSet"] == "train"]
    X_tr = (
        df_train.drop(["imageSet", "category", "imageFileName"], axis=1)
        .astype(float)
        .values
    )
    y_tr = df_train.category.values

    scaler = StandardScaler()
    X_tr_rescaled = scaler.fit_transform(X_tr)

    pca = PCA(n_components=2)
    pca.fit(X_tr_rescaled, y_tr)
    X_2d = pca.transform(X_tr_rescaled)

    # let's plot the 2 dimensional representation of our data using the first 2 components
    # for each category
    fig = plt.figure()
    for cat in categories:
        idx = y_tr == cat

        plt.scatter(X_2d[idx, 0], X_2d[idx, 1], label="type {}".format(cat))

    plt.legend()
    plt.xlabel("1st component")
    plt.ylabel("2nd component")
    plt.savefig("./output/PCA.png")


def main():

    print("------------------------------------------------")
    print(" DATA EXPLORATION")

    """  Goal:
        # Start by some quick data exploration.
        # 
        # Plot a few images from each category from the test dataset
        # Visualize the features on a 2d-plot with PCA using the first two components.
        """

    # first load the dataset
    with np.load("./output/images_data.npz", allow_pickle=True) as npz_file:
        df = pd.DataFrame(npz_file["values"], columns=npz_file["columns"])

    # plot some images of each categories from test dataset
    categories = ["bike", "car", "motorcycle", "other", "truck", "van"]
    df_test = df[df["imageSet"] == "test"]
    plot_images(df_test, categories)
    print("     - Images examples : check plot under./output/images_display.png")

    # let's apply a Principal component analysis to evidence main components
    apply_PCA(df, categories)
    print("     - PCA plot : check plot under./output/PCA.png")
    print("------------------------------------------------")

    return df
