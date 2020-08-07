import cv2 as cv
import utilis
import numpy as np
import glob
import csv

# mask = cv.imread("./Data/Fluo-N2DL-HeLa/Sequence_1_masks/t012mask.tif", -1)

# mask = cv.imread("./Data/PhC-C2DL-PSC/Sequence_1_Masks/t122mask.tif", -1)

masks = [f for f in glob.glob("./Data/Fluo-N2DL-HeLa/Sequence_2_masks/*")] 

mask = cv.imread(masks[0], -1)
mask, _, _ = utilis.stretch(mask.astype(np.uint8))

print(masks[0])

cv.imshow(masks[0], mask)
cv.waitKey(0)
cv.destroyAllWindows()

# row_list = [["Filename", "Count"]]

# for mask_path in masks:
#     mask = cv.imread(mask_path, -1)

#     _, cnt, _ = cv.findContours(mask.astype(np.uint8), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

#     # cv.imshow("mask", mask)
    
#     _, name = utilis.pathname(mask_path)
#     row_list.append([name, len(cnt)])

# with open('Fluo_results2.csv', 'w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerows(row_list)