## import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2 as cv

from scipy import ndimage as ndi
from skimage import color
from skimage.future import graph
from sklearn.cluster import MeanShift, estimate_bandwidth

from skimage.segmentation import watershed, quickshift, mark_boundaries, slic
from skimage.feature import peak_local_max
from skimage.util import img_as_float

from PIL import Image
import argparse

from utilis import *

parser = argparse.ArgumentParser(description='Meanshift implementation')
parser.add_argument('file', type=str, nargs=1, 
    help='Path to file')
parser.add_argument('--bandwidth', type=float, nargs="?", default=None, 
    help='bandwidth for meanshift')

args = parser.parse_args()
filename = args.file[0]

size = 512, 512

# img_names = ["coins.png", "kiwi.png"]
# ext_names = ["t000.tif", "t010.tif"]

# images = [i for i in img_names]
# ext_images = [i for i in ext_names]


def meanshift(img_mat):
    colour_samples = np.reshape(img_mat, [-1,3])

    bandwidth = args.bandwidth

    if args.bandwidth is None:
        bandwidth = estimate_bandwidth(colour_samples, quantile=0.2, n_samples=500)

    print(bandwidth)

    ms_clf = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms_clf.fit(colour_samples)

    ms_labels = ms_clf.labels_

    cluster_centers = ms_clf.cluster_centers_ 

    # Step 4 - reshape ms_labels back to the original image shape 
    # for displaying the segmentation output 
    ms_labels = np.reshape(ms_labels, img_mat.shape[:2])

    print(f"shape: {ms_labels.shape} n_clusters: {len(np.unique(ms_labels))}")

    # print(np.unique(ms_labels))

    mappings = dict.fromkeys(np.unique(ms_labels), 0)

    aggregate(mappings, ms_labels)

    # print(mappings)

    segmentedImg = cluster_centers[ms_labels]

    print(f"cluster_centers.shape: {cluster_centers.shape}, ms_labels: {ms_labels.shape}, segmentedImg: {segmentedImg.shape}")
    # print(segmentedImg)
    # exit(1)

    # cv.imshow("Cropped", np.uint8(segmentedImg))
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    return ms_labels, mappings

def _watershed(img_mat):
    img_array = cv.cvtColor(img_mat, cv.COLOR_BGR2GRAY)

    # Step 2 - Calculate the distance transform
    distance = ndi.distance_transform_edt(img_array)

    # Step 3 - Generate the watershed markers
    local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((8, 8)),
                        labels=img_array)

    markers = ndi.label(local_maxi)[0]

    # Step 4 - Perform watershed and store the labels
    ws_labels = watershed(distance, markers, mask=img_array)

    # Display the results
    # plot_three_images(filename, img, "Original Image", ms_labels, "MeanShift Labels",
    #                     ws_labels, "Watershed Labels")

    return ws_labels

# print(f"Reading {filename}")

img = Image.open(filename)
img.thumbnail(size)

img = np.array(img)[:, :, :3]
img = stretch(img)

plt.imshow(~img)
plt.axis('off')
plt.title("Initial")
plt.show()

ws_labels = _watershed(~img)

plot_two("Watershed", img, "OG", ws_labels, "Watershed")
exit(1)

img_mat = img[:, :, :3] #Copy the pixels (no need for :3 since no colour channels??)

ms_labels, mappings = meanshift(img_mat)

ms_g = graph.rag_mean_color(img_mat, ms_labels)

ms_labels2 = graph.merge_hierarchical(ms_labels, ms_g, thresh=35, rag_copy=False,
                                   in_place_merge=True,
                                   merge_func=merge_mean_color,
                                   weight_func=weight_mean_color)

ms_out = color.label2rgb(ms_labels2, img_mat, kind='avg', bg_label=0)

ms_results = color.label2rgb(ms_labels, img_mat, kind='avg', bg_label=0)

plt.imshow(ms_results)
plt.axis('off')
plt.title("Meanshift results")
plt.show()

# def remove_clusters():
for cluster in reversed(list(mappings.keys())):
    set_label(cluster, ms_labels, 0)

    ms_out = color.label2rgb(ms_labels, img_mat, kind='avg', bg_label=0)

    plt.imshow(ms_out)
    plt.axis('off')
    plt.title(f"Absolved cluster: {cluster}")
    plt.show()

ws_labels = _watershed(img_mat)


plot_three_images("Meanshift vs Watershed", img_mat, "Original", 
    ms_results, "Menshift", ws_labels, "Watershed")

# Quickshift

qs_labels = quickshift(img_mat, kernel_size=3, max_dist=5, ratio=0.5)
qs_g = graph.rag_mean_color(img_mat, qs_labels)

qs_labels2 = graph.merge_hierarchical(qs_labels, qs_g, thresh=35, rag_copy=False,
                                   in_place_merge=True,
                                   merge_func=merge_mean_color,
                                   weight_func=weight_mean_color)

qs_out = color.label2rgb(qs_labels2, img_mat, kind='avg', bg_label=0)
qs_out = mark_boundaries(qs_out, qs_labels2, (0, 0, 0))

plot_three_images("Meanshift vs Quichshift", img_mat, "Original", 
    ms_results, "Menshift", qs_out, "Quickshift")