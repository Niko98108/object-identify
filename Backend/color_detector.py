from __future__ import print_function
import binascii
from PIL import Image
import numpy as np
import scipy
import scipy.misc
import scipy.cluster
from math import sqrt

NUM_CLUSTERS = 5
COLORS = (
    (244, 67, 54),  # red
    (255, 235, 59),  # yellow
    (255, 152, 0),  # orange / you can add as many color here

)


def closest_color(rgb):
    r, g, b = rgb
    color_diffs = []
    for color in COLORS:
        cr, cg, cb = color
        color_diff = sqrt((r - cr) ** 2 + (g - cg) ** 2 + (b - cb) ** 2)
        color_diffs.append((color_diff, color))
    return min(color_diffs)[1]


def get_dominant_color(path):
    im = Image.open(path)
    im = im.resize((150, 150))  # optional, to reduce time
    ar = np.asarray(im)
    shape = ar.shape
    ar = ar.reshape(scipy.product(shape[:2]), shape[2]).astype(float)

    codes, dist = scipy.cluster.vq.kmeans(ar, NUM_CLUSTERS)
    print('cluster centres:\n', codes)

    vecs, dist = scipy.cluster.vq.vq(ar, codes)  # assign codes
    counts, bins = scipy.histogram(vecs, len(codes))  # count occurrences

    index_max = scipy.argmax(counts)  # find most frequent
    peak = codes[index_max]
    colour = binascii.hexlify(bytearray(int(c) for c in peak)).decode('ascii')
    rgb_color = tuple(int(colour[i:i + 2], 16) for i in (0, 2, 4))

    cl_color = closest_color(rgb_color)

    return cl_color


#print(get_dominant_color('Test/test_image2.png'))
