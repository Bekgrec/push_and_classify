import numpy
import cv2
import pandas as pd

dep_img = cv2.imread('need_check/depth_image_000000.png')
print(dep_img.shape)
d1 = dep_img[:, :, 0]
print(d1.shape)
d1_df = pd.DataFrame(d1)
d1_df.to_csv('d1_csv')
d2 = dep_img[:, :, 1]
d3 = dep_img[:, :, 2]
a = (d1 == d2)
print(sum(sum(a)))
b = (d2 == d3)
print(sum(sum(b)))
print(dep_img[0 ,0 ,0])
