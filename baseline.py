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
# ## Goal:
# A strong baseline to evaluate is logistic regression
# 
# Evaluate the logistic regression baseline
# Tune its regularization strength parameter with cross-validated grid-search
# Pick ten images and compute the probability for each class using the predict_proba() function of your estimator. Plot the results for each image and comment your findings. For example, you can plot the ten images in subplots and collect the probabilities in a DataFrame.
# %% [markdown]
# ### load the npz file and create the array to be then used for model training and testing

# %%
with np.load('images_data.npz', allow_pickle=True) as npz_file:
    df = pd.DataFrame(npz_file['values'], columns= npz_file['columns'])
    
df_train = df[df['imageSet'] == 'train']
df_test = df[df['imageSet'] == 'test']

X_tr = df_train.drop(['imageSet','category','imageFileName'], axis=1).values
X_tr = np.float32(X_tr)

X_te = df_test.drop(['imageSet','category','imageFileName'], axis=1).values
X_te = np.float32(X_te)

features_names=df_train.drop(['imageSet','category','imageFileName'], axis=1).columns
categories = ['bike', 'car', 'motorcycle','other','truck','van']

# %% [markdown]
# ### transform the categories into numerical values to be then used for logistic regression

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
df_test['category_num'] = df_test['category'].apply(cat2num)
y_tr = df_train.category_num.values
y_te = df_test.category_num.values

# %% [markdown]
# ### fit a multi-class logistic regression with OvR:
# - let's create a pipeline with a first step as standscaler

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('logreg', LogisticRegression(C=1, multi_class='ovr', solver='liblinear', random_state=0))
])

pipe.fit(X_tr, y_tr)

accuracy = pipe.score(X_te, y_te)
print('Accuracy: {:.3f}'.format(accuracy))

# %% [markdown]
# - we obtain an accuracy of 92% without tuning the strength parameter
# %% [markdown]
# ### Apply a gridsearchCV to improve results of our logistic regression:
# - shuffle the data to improve (maybe) the results...

# %%
from sklearn.utils import shuffle
X_tr_s, y_tr_s = shuffle(X_tr, y_tr)


# %%
from sklearn.model_selection import GridSearchCV

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('logreg', LogisticRegression(multi_class='ovr', solver='liblinear', random_state=0))
])
grid = {'logreg__C': np.linspace(0.0001, 1000, 10)}
grid_cv = GridSearchCV(pipe, grid, cv=5, return_train_score=True)

grid_cv.fit(X_tr_s, y_tr_s)


# %%
cv_results = pd.DataFrame(grid_cv.cv_results_)

cols = ['mean_test_score', 'std_test_score', 'mean_train_score', 'std_train_score', 'param_logreg__C']
cv_results[cols].sort_values('mean_test_score', ascending=False)


# %%
grid_cv.score(X_te, y_te)

# %% [markdown]
# - I get  94% accuracy with cross validation.

# %%
ind = np.random.choice(len(X_tr)-1, 10)


# %%
proba_results = []

# Create a grid of subplots
fig, axes = plt.subplots(nrows=1, ncols=10, figsize=(30, 30))

# Plot the 10 nearest images 
for i, axis in enumerate(axes.flatten()):
    img = Image.open(df_train['imageFileName'].iloc[ind[i]])
    title = df_train['imageFileName'].iloc[ind[i]].split('\\')[1]
    axis.set_title(title)
    proba = grid_cv.predict_proba(X_tr[ind[i]].reshape(1,-1))
    
    proba_results.append({
            'name': title,
            'bike': proba[0][0],
            'car': proba[0][1],
            'motorcycle': proba[0][2],
            'other': proba[0][3],
            'truck': proba[0][4],
            'van': proba [0][5]       
    })
        

    # Plot kernel with imshow()    
    axis.imshow(img)
    axis.get_xaxis().set_visible(False)
    axis.get_yaxis().set_visible(False)
    
plt.show()


# %%
# Convert results to DataFrame
proba_results = pd.DataFrame(proba_results)
proba_results.set_index('name', inplace=True)
proba_results.round(2)

# %% [markdown]
# - the logistic regression is giving the higher probability to the corrisponding category for our 10 exemples
# - the more we increase the regularization strength the highest is the probability of the corrisponding category until it reaches 1.
# - the more we decrease the regularization strength the less the probability prediction is "stark" identified
# - this is due to the sigmoid function which associate large  positive and negative values with probabilities close to respectively 1 and 0 and 0.5 for x=0
#     - indeed increasing C, reduces penaliation and therefore large coefficients will lead to probabilities close to 1 or 0
#     - on the contrary decreasing C increases the penalization and therefore the sigmoid value, probability will converge toward 1/6

# %%
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('logreg', LogisticRegression(C=0.0001, multi_class='ovr', solver='liblinear', random_state=0))
])
pipe.fit(X_tr_s,y_tr_s)
pipe.score(X_te,y_te)


# %%



