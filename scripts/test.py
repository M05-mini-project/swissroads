import numpy as np
import pandas as pd
import collections
import os

import database
import baseline

# baseline tests
def function_cat2num(input, expected):
  output = baseline.cat2num(input)
  assert output == expected, 'Expected %r, but got %r' % (expected, output)

def test_cat2num_1():
  function_cat2num('bike', 0)

def test_cat2num_2():
  function_cat2num('other', 3)


# database tests
TEST_SIZE = 50
TRAIN_SIZE = 280
VALID_SIZE = 139

def test_getimage_wrongFolder():
    batches_data, batches_cat, batches_file, batches_folder = database.get_images('../fake_folder')

    assert batches_data == [], 'Expected %r, but got %r' % ([], batches_data)
    assert batches_cat == [], 'Expected %r, but got %r' % ([], batches_cat)
    assert batches_file == [], 'Expected %r, but got %r' % ([], batches_file)
    assert batches_folder == [], 'Expected %r, but got %r' % ([], batches_folder)

def test_getimage():
    batches_data, batches_cat, batches_file, batches_folder = database.get_images('../swissroads_images')

    #batches lenght
    data_lenght = 469
    assert len(batches_data) == data_lenght, 'Expected %r, but got %r' % ([], len(batches_data))
    assert len(batches_cat) == data_lenght, 'Expected %r, but got %r' % ([], len(batches_cat))
    assert len(batches_file) == data_lenght, 'Expected %r, but got %r' % ([], len(batches_file))
    assert len(batches_folder) == data_lenght, 'Expected %r, but got %r' % ([], len(batches_folder))

    #label correspond to image name
    for i in range(len(batches_cat)):
        assert batches_cat[i] in batches_file[i], 'Expected to contain %r, but got %r' % (batches_cat[i], batches_file[i])

    #batches_folder size
    counter = collections.Counter(batches_folder)

    assert counter['test'] == TEST_SIZE, 'Expected %r, but got %r' % (TEST_SIZE, counter['test'])
    assert counter['train'] == TRAIN_SIZE, 'Expected %r, but got %r' % (TRAIN_SIZE, counter['train'])
    assert counter['valid'] == VALID_SIZE, 'Expected %r, but got %r' % (VALID_SIZE, counter['valid'])

def test_get_batches():
    list_lenght = 1300 #arbitrary number, can work with any int
    batch_size = 32

    test_list = np.arange(list_lenght)

    expected_lengh = batch_size

    for id, batch in enumerate(database.get_batches(test_list, batch_size)):
        if id == int(list_lenght/batch_size):
            expected_lengh = list_lenght%batch_size

        assert len(batch)==expected_lengh, 'Expected %r, but got %r' % (expected_lengh, len(batch))
        assert batch[0] == test_list[id*batch_size], 'Expected %r, but got %r' % (test_list[id*batch_size], batch[0])

def test_load_database():
    #create local folder
    output_path = r'./output' 
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    folder_name = '../swissroads_images'
    database.load_data(folder_name)

    database_path = './output/images_data.npz'

    assert os.path.isfile(database_path), 'File %r does not exist' %(database_path)

    #assert generated content
    with np.load('./output/images_data.npz', allow_pickle=True) as npz_file:
        df = pd.DataFrame(npz_file['values'], columns= npz_file['columns'])
    df_test = df[df['imageSet'] == 'test']
    df_train = df[df['imageSet'] == 'train']
    df_valid = df[df['imageSet'] == 'valid']

    assert len(df_test) == TEST_SIZE, 'Expected %r, but got %r' % (TEST_SIZE, len(df_test))
    assert len(df_train) == TRAIN_SIZE, 'Expected %r, but got %r' % (TRAIN_SIZE, len(df_train))
    assert len(df_valid) == VALID_SIZE, 'Expected %r, but got %r' % (VALID_SIZE, len(df_valid))

    categories = ['bike', 'car', 'motorcycle','other','truck','van']

    for category in categories:
        path = '../swissroads_images/test/' + category
        list = os.listdir(path)

        df_categories = df_test[df_test['category'] == category]
        assert len(df_categories) == len(list), 'Expected %r files, but got %r' % (len(list), len(df_categories))

    #delete local folder
    os.remove(database_path)
    os.rmdir(output_path)

test_load_database()
