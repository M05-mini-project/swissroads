import database
#import data_exploration
#import baseline
#import analysis

database.load_data()

import pandas as pd

model_results = [{'model': 'logistic','test_accuracy': 0.94},
                {'model': '1-layer nn','test_accuracy': 0.9},
                {'model': '2-layer nn','test_accuracy': 0.96}]


# Convert results to DataFrame
model_results = pd.DataFrame(model_results)
model_results


import matplotlib.pyplot as plt

plt.figure(figsize=(16,8))
plt.bar(model_results.model, model_results.test_accuracy)
plt.show()

# - The CNN is not performing that well as the previous algo. the amount of pictures to learn is quite low , 280 images for 6 classes. 
#Moreover only 2 convolution layers were applied and one hidden layer on the connected layer which I guess is less than the CNN Mobilenet V2 to obtain the 1280 features.

