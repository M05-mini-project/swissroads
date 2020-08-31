# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
import database
import preprocessor
import algorithm
import analysis

import pandas as pd

model_results = [{'model': 'k-NN','test_accuracy': 0.96},
                 {'model': 'decision Tree','test_accuracy': 0.92},
                {'model': 'logistic','test_accuracy': 0.94},
                {'model': 'random forest','test_accuracy': 0.96},
                {'model': 'svm linear','test_accuracy': 0.90},
                {'model': 'svm rbf','test_accuracy': 0.94},
                {'model': '1-layer nn','test_accuracy': 0.9},
                {'model': '2-layer nn','test_accuracy': 0.96},
                {'model': 'cnn','test_accuracy': 0.68},]


# Convert results to DataFrame
model_results = pd.DataFrame(model_results)
model_results


# %%
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

plt.figure(figsize=(16,8))
plt.bar(model_results.model, model_results.test_accuracy)

# %% [markdown]
# - The CNN is not performing that well as the previous algo. the amount of pictures to learn is quite low , 280 images for 6 classes. Moreover only 2 convolution layers were applied and one hidden layer on the connected layer which I guess is less than the CNN Mobilenet V2 to obtain the 1280 features.

# %%



