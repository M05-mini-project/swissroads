from baseline import cat2num

def test_cat2num(input, expected):
  output = cat2num(input)
  assert output == expected, 'Expected %r, but got %r' % (expected, cer)

def test_cat2num_1():
  test_cat2num('bike', 0)

def test_cat2num_2():
  test_cat2num('other', 3)