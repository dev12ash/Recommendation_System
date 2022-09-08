import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

# skimage apis for image manipulations
from skimage.io import imread, imshow
from skimage.color import rgb2gray, rgb2hsv
from skimage.transform import resize, rescale
from skimage.transform import rotate
from skimage import exposure
from skimage.filters import sobel_h, sobel_v
from skimage.filters import prewitt_h, prewitt_v

# image similarity and classification
from keras.datasets import fashion_mnist
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import manhattan_distances
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier

# import metrics
from sklearn.metrics import confusion_matrix, accuracy_score

# read the image using skimage.imread and plot using imshow function
building_img = imread("/kaggle/input/vyronas-database/Vyronasdbmin/building44_front_day.jpg",
                      as_gray=False)
print("Shape of image is: ", building_img.shape)
print(building_img)
imshow(building_img)

# rescaling image sing rescale function of skimage.rescale
build_rescale = rescale(building_img, scale=(0.6,0.5,1/3))
print("Shape of rescaled image: ", build_rescale.shape)
imshow(build_rescale)

# rotate the image using rotate method of skimage
rotate_img = rotate(building_img, angle=60)
print("Rotate image shape same as original: ", rotate_img.shape)
imshow(rotate_img)

# flip the images left to right and up to down using numpy's fliplr and flipud
build_lr = np.fliplr(building_img)
build_ud = np.flipud(building_img)
plt.figure(figsize=(10,8))
plt.subplot(1,3,1)
imshow(building_img)
plt.title("Original RGB format")
plt.subplot(1,3,2)
imshow(build_lr)
plt.title("left to right flip")
plt.subplot(1,3,3)
imshow(build_ud)
plt.title("Up to down flip ")

# adjusting brightness of the images
bright = exposure.adjust_gamma(building_img, gamma=0.20, gain=1)
dark = exposure.adjust_gamma(building_img, gamma=3, gain=1)
darkest = exposure.adjust_gamma(building_img, gamma=8, gain=1)

# plot the images
plt.figure(1, figsize=(10,8))
plt.subplot(121)
imshow(building_img)
plt.title("Original RGB format")
plt.subplot(122)
imshow(bright)
plt.title("bright img with gamma=0.20")

# cropping images 
cropped_image1 = building_img[50: (building_img.shape[0]-100), 50: (building_img.shape[1]-150)]
imshow(cropped_image1)