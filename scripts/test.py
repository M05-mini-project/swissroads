import numpy as np
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
    test_size = 50
    train_size = 280
    valid_size = 139

    counter = collections.Counter(batches_folder)

    assert counter['test'] == test_size, 'Expected %r, but got %r' % (train_size, counter['test'])
    assert counter['train'] == train_size, 'Expected %r, but got %r' % (train_size, counter['train'])
    assert counter['valid'] == valid_size, 'Expected %r, but got %r' % (train_size, counter['valid'])

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
    output_path = r'./output' 
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    folder_name = '../swissroads_images'
    database.load_data(folder_name)

    database_path = './output/images_data.npz'

    assert os.path.isfile(database_path), 'File does not exist'

    os.remove(database_path)
    os.rmdir(output_path)

test_load_database()
