import os
import cv2
import glob
import random
import numpy as np

img_path = 'avatars'
full_pathes = glob.glob( os.path.join( img_path, '*png' ) )

images = []
for idx, image_path in enumerate(full_pathes):
    images.append(cv2.imread(image_path, 1))

height, width, channel = images[0].shape
max_intensity = 256

intensity = np.zeros([*images[0].shape, max_intensity])
for image in images:
    for h in range(height):
        for w in range(width):
            for c in range(channel):
                intensity[h, w, c, int(image[h, w, c])] += 1

proba = intensity / len(full_pathes)

result_image = np.zeros(images[0].shape)
for h in range(height):
    for w in range(width):
        for c in range(channel):
            result_image[h, w, c] = random.choices(np.arange(max_intensity), proba[h, w, c])[0]

cv2.imshow('', result_image.astype(np.uint8))
cv2.waitKey(0)