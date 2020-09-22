from scripts import database
from scripts import baseline

def function_cat2num(input, expected):
  output = baseline.cat2num(input)
  assert output == expected, 'Expected %r, but got %r' % (expected, cer)

def test_cat2num_1():
  function_cat2num('bike', 0)

def test_cat2num_2():
  function_cat2num('other', 3)


print(database.get_images('fake name'))
print(database.get_images('../swissroads_images'))
