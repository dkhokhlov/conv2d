import numpy as np
import conv

print('\n======================== Test 1')
image = np.array(
    [[[
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ]]], dtype='float32')  # N=1, Ci=1, H=4, W=4
kernel = np.ones((3, 1, 3, 3), dtype='float32') # Co, Ci, K, K
output = np.zeros((1, 3, 4, 4), dtype='float32') # N, Co, H, W
conv.conv2d(image, kernel, output)
print(image)
print('------')
print(kernel)
print('------')
print(output)

print('\n======================== Test 2')
image = np.array(
    [[[
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1]
    ]]], dtype='float32')  # N=1, Ci=1, H=4, W=4
kernel = np.ones((1, 1, 1, 1), dtype='float32') # Co, Ci, K, K
output = np.zeros((1, 1, 4, 4), dtype='float32') # N, Co, H, W
conv.conv2d(image, kernel, output)
print(image)
print('------')
print(kernel)
print('------')
print(output)

print('\n======================== Test 3')
kernel = np.ones((1, 1, 1, 1), dtype='float32') # Co, Ci, K, K
output = np.zeros((1, 1, 4, 4), dtype='float32') # N, Co, H, W
conv.conv2d(image, kernel, output)
print(image)
print('------')
print(kernel)
print('------')
print(output)

print('\n======================== Test 4')
image = np.array(
    [[[
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 10, 11],
        [12, 13, 14, 15]
    ]]], dtype='float32')  # N=1, Ci=1, H=4, W=4
kernel = np.ones((3, 1, 3, 3), dtype='float32') # Co, Ci, K, K
output = np.zeros((1, 3, 4, 4), dtype='float32') # N, Co, H, W
conv.conv2d(image, kernel, output, (0., 0.01))
print(image)
print('------')
print(kernel)
print('------')
print(output)
print('------')
print('bias', (0., 0.01))

