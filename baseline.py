import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import PIL.Image as Image

	
pd.options.mode.chained_assignment = None 

#### transform the categories into numerical values to be then used for logistic regression
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

### Goal:
# A strong baseline to evaluate is logistic regression
# 
# Evaluate the logistic regression baseline
# Tune its regularization strength parameter with cross-validated grid-search
# Pick ten images and compute the probability for each class using the predict_proba() function of your estimator. Plot the results for each image and comment your findings. For example, you can plot the ten images in subplots and collect the probabilities in a DataFrame.

def main(df):
    df_train = df[df['imageSet'] == 'train']
    df_test = df[df['imageSet'] == 'test']

    X_tr = df_train.drop(['imageSet','category','imageFileName'], axis=1).values
    X_tr = np.float32(X_tr)

    X_te = df_test.drop(['imageSet','category','imageFileName'], axis=1).values
    X_te = np.float32(X_te)

    features_names=df_train.drop(['imageSet','category','imageFileName'], axis=1).columns
    categories = ['bike', 'car', 'motorcycle','other','truck','van']

    
    df_train['category_num'] = df_train['category'].apply(cat2num)
    df_test['category_num'] = df_test['category'].apply(cat2num)
    y_tr = df_train.category_num.values
    y_te = df_test.category_num.values

    # ### fit a multi-class logistic regression with OvR:
    # - let's create a pipeline with a first step as standscaler
    print('---------------------------------------------------')
    print('BASELINE : optimizing logistic regression model...')
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.utils import shuffle
    from sklearn.model_selection import GridSearchCV

    # shuffle the data
    X_tr_s, y_tr_s = shuffle(X_tr, y_tr, random_state=0)

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('logreg', LogisticRegression(multi_class='ovr', solver='liblinear', random_state=0))
    ])

    # perform a gridsearch with our pipeline
    grid = {'logreg__C': np.linspace(0.0001, 1000, 10)}
    grid_cv = GridSearchCV(pipe, grid, cv=5, return_train_score=True)
    grid_cv.fit(X_tr_s, y_tr_s)

    # store the results
    cv_results = pd.DataFrame(grid_cv.cv_results_)
    cols = ['mean_test_score', 'std_test_score', 'mean_train_score', 'std_train_score', 'param_logreg__C']
    cv_results[cols].sort_values('mean_test_score', ascending=False)

    baseline_acc_tr = grid_cv.score(X_tr_s, y_tr_s)
    baseline_acc_te = grid_cv.score(X_te, y_te)

    print('------------------------------------------------')
    print(' baseline - logistic regression : ')
    print(' train accuracy : ', baseline_acc_tr)
    print(' test accuracy : ', baseline_acc_te)
    print('------------------------------------------------')

    #confusion matrix
    from sklearn.metrics import confusion_matrix

    cm =confusion_matrix(y_te, grid_cv.predict(X_te), labels=[0,1,2,3,4,5])
    df_res = pd.DataFrame(cm, columns=['bike','car','motorcycle','other','truck','van'])
    df_res['label'] = ['bike','car','motorcycle','other','truck','van']
    df_res.set_index('label', inplace=True)

    print('Baseline confusion matrix:')
    print(df_res)
    print('------------------------------------------------')

    return baseline_acc_tr, baseline_acc_te

