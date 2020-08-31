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
# Start by some quick data exploration. Perform the following tasks
# 
# Plot a few images from each category
# Make a scree plot, how many components explain 10%, 20%, …, 90% and 100% of the variance?
# Visualize the features on a 2d-plot with PCA using the first two components. What can you say about the results?
# %% [markdown]
# # load the previous section .npz file with high level features

# %%
with np.load('images_data.npz', allow_pickle=True) as npz_file:
    print(list(npz_file.keys()))
    print('features:', npz_file['columns'])   
    df = pd.DataFrame(npz_file['values'], columns= npz_file['columns'])

print(df.shape)
df.head()

# %% [markdown]
# # plot images from valid dataset

# %%
# plot some images of each categories from test dataset
categories = ['bike', 'car', 'motorcycle','other','truck','van']
df_test = df[df['imageSet'] == 'test']

# Create a grid of subplots
fig, axes = plt.subplots(nrows=6, ncols=3, figsize=(10, 10))

# Remove gaps between suplots
plt.subplots_adjust(wspace=0, hspace=0)    

for i in range(len(categories)):
    df2 = df_test[df_test['category'] == categories[i]]
    for j in range(3):
        axis = axes.flatten()[3*i+j]
        img = Image.open(df2['imageFileName'].iloc[j]) 

        axis.imshow(img)
        axis.get_xaxis().set_visible(False)
        axis.get_yaxis().set_visible(False)
    
plt.show()
    

# %% [markdown]
# - we get 3 image samples from our 6 different categpries
# %% [markdown]
# # let's apply a Principal component analysis to evidence main components
# - scale the data
# - apply a PCA with 2 components on train dataset
# 

# %%
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

df_train = df[df['imageSet'] == 'train']
X_tr = df_train.drop(['imageSet','category','imageFileName'], axis=1).values
y_tr = df_train.category.values

scaler = StandardScaler()
X_tr_rescaled = scaler.fit_transform(X_tr)

pca = PCA(n_components=2)
pca.fit(X_tr_rescaled, y_tr)
X_2d = pca.transform(X_tr_rescaled)

# %% [markdown]
# # let's plot the 2 dimensional representation of our data using the first 2 components

# %%
# Plot each category
for cat in categories:
    idx = (y_tr == cat)

    plt.scatter(
        X_2d[idx, 0], X_2d[idx, 1],
        label='type {}'.format(cat)
    )

plt.legend()
plt.xlabel('1st component')
plt.ylabel('2nd component')
plt.show()

# %% [markdown]
# ### what we notice:
# - we see clearly 6 clusters representing our 6 categories
# - Truck and van category are in the same area when car, bike and motorcycle are clearly grouped differently
# - the 'other' category is bright and as 'clearly' regrouped as the other categories

# %%
pca.explained_variance_ratio_.sum()

# %% [markdown]
# - with 2 components we obtain around 15% of the variance
# ### let's perform a pca without specifying the amount of components

# %%
# Create PCA transformer
pca_all = PCA(n_components=None)

# Apply PCA to rescaled data
pca_all.fit(X_tr_rescaled)

# Proportion of variance explained
pve = pca_all.explained_variance_ratio_
pve.shape

# %% [markdown]
# - we obtain 280 components when not specifying the components to the PCA
# ### let's plot the cumulative proportion of variance explained

# %%
# Create bar plot
xcor = np.arange(1, len(pve) + 1)
plt.bar(xcor, pve)
plt.xticks(xcor)

# Add cumulative sum
pve_cumsum = np.cumsum(pve)
plt.step( xcor+0.5, pve_cumsum,  label='cumulative')

# Add labels
plt.xlabel('principal component')
plt.ylabel('proportion of variance explained')
plt.legend()
plt.show()


# %%
for i in range(10, 101, 10):
    print(str(np.where(pve_cumsum > i/100)[0][0]+1) + ' component(s) explain(s) ' + str(i) + '% of the variance')
    

# %% [markdown]
# - 100% of the variance is ontained with 279 components

# %%



