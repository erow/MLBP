# preprossessing for mnist http://yann.lecun.com/exdb/mnist/
# used library : https://docs.python.org/3/library/struct.html
from struct import unpack_from
from torch import tensor

mean = 0
var = 0.04
#
# The labels values are 0 to 9.
# TRAINING SET IMAGE FILE (train-images-idx3-ubyte):
# [offset] [type]          [value]          [description]
# 0000     32 bit integer  0x00000803(2051) magic number
# 0004     32 bit integer  60000            number of images
# 0008     32 bit integer  28               number of rows
# 0012     32 bit integer  28               number of columns
# 0016     unsigned byte   ??               pixel
# 0017     unsigned byte   ??               pixel
# ........
# xxxx     unsigned byte   ??               pixel
def training_set_image():
    with open('data/raw/train-images-idx3-ubyte', 'rb') as f:
        (magic_num, num, rows, cols) = unpack_from('>4i', f.read(16))
        assert 2051 == magic_num, magic_num
        print(num, rows, cols)
        images = []
        for c in range(num):
            pixels = unpack_from(str(rows * cols) + 'B', f.read(rows * cols))
            images.append(list(pixels))
        t = (tensor(images).float()/255.0-mean)/var
        # (C, H, W)
        return t.view(num, 1, cols, rows)


#  TRAINING SET LABEL FILE (train-labels-idx1-ubyte):
# [offset] [type]          [value]          [description]
# 0000     32 bit integer  0x00000801(2049) magic number (MSB first)
# 0004     32 bit integer  60000            number of items
# 0008     unsigned byte   ??               label
# 0009     unsigned byte   ??               label
# ........
# xxxx     unsigned byte   ??               label
def training_set_label():
    with open('data/raw/train-labels-idx1-ubyte', 'rb') as f:
        (magic_num, num) = unpack_from('>2i', f.read(8))
        assert 2049 == magic_num, magic_num
        labels = unpack_from(str(num) + 'B', f.read(num))
        return tensor(list(labels)).view(num)


#  TEST SET LABEL FILE (t10k-labels-idx1-ubyte):
# [offset] [type]          [value]          [description]
# 0000     32 bit integer  0x00000801(2049) magic number (MSB first)
# 0004     32 bit integer  10000            number of items
# 0008     unsigned byte   ??               label
# 0009     unsigned byte   ??               label
# ........
# xxxx     unsigned byte   ??               label
def test_set_label():
    with open('data/raw/t10k-labels-idx1-ubyte', 'rb') as f:
        (magic_num, num) = unpack_from('>2i', f.read(8))
        assert 2049 == magic_num, magic_num
        labels = unpack_from(str(num) + 'B', f.read(num))
        return tensor(list(labels)).view(num)


# The labels values are 0 to 9.
# TEST SET IMAGE FILE (t10k-images-idx3-ubyte):
# [offset] [type]          [value]          [description]
# 0000     32 bit integer  0x00000803(2051) magic number
# 0004     32 bit integer  10000            number of images
# 0008     32 bit integer  28               number of rows
# 0012     32 bit integer  28               number of columns
# 0016     unsigned byte   ??               pixel
# 0017     unsigned byte   ??               pixel
# ........
# xxxx     unsigned byte   ??               pixel
def test_set_image():
    with open('data/raw/t10k-images-idx3-ubyte', 'rb') as f:
        (magic_num, num, rows, cols) = unpack_from('>4i', f.read(16))
        assert 2051 == magic_num, magic_num
        print(num, rows, cols)
        images = []
        for c in range(num):
            pixels = unpack_from(str(rows * cols) + 'B', f.read(rows * cols))
            images.append(list(pixels))
        t = (tensor(images).float()/255.0-mean)/var
        # (C, H, W)
        return t.view(num, 1, cols, rows)

if __name__ == "__main__":
    print('ok')
    print(training_set_label())
