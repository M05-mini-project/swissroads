import numpy as np
import pandas as pd

from baseline import cat2num
import analysis

def function_cat2num(input, expected):
  output = cat2num(input)
  assert output == expected, 'Expected %r, but got %r' % (expected, cer)

def test_cat2num_1():
  function_cat2num('bike', 0)

def test_cat2num_2():
  function_cat2num('other', 3)

def test_analysis_create_data_sets():

  # create a dataFrame and check the data split 
  data = {'data1':[20, 21, 19, 18],
          'data2':[20, 21, 19, 18],
          'imageSet':['train', 'train', 'test', 'valid'],
          'category':['bike','car','van','truck'],
          'imageFileName':['file1','file2','file3','file4'],
          'category_num':[0,1,2,3],
    } 
  
  # Create DataFrame 
  df = pd.DataFrame(data) 
  
  X_tr, y_tr, X_te, y_te, X_va, y_va = analysis.create_data_sets(df)

  assert len(y_tr) == 2, 'Expected %r, but got %r' % (2, len(y_tr))
  assert len(y_te) == 1, 'Expected %r, but got %r' % (1, len(y_te))
  assert len(y_va) == 1, 'Expected %r, but got %r' % (1, len(y_va))
  assert X_tr.shape == (2,3), 'Expected %r, but got %r' % ((2,3), X_tr.shape)