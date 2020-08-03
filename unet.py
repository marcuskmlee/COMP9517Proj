import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from PIL import Image
from torch.nn.functional import sigmoid
import matplotlib.pyplot as plt
import argparse
import torchvision.transforms as T
import cv2 as cv

from unet_utils import *

parser = argparse.ArgumentParser(description='Meanshift implementation')
parser.add_argument('file', type=str, nargs=1, 
    help='Path to file')

args = parser.parse_args()
filename = args.file[0]

device = "cpu"
model_path = "./Unet/model_epoch_2290.pwf"
img = cv.imread(filename)

net = torch.load(model_path, map_location=torch.device('cpu'))
if isinstance(net,torch.nn.DataParallel):
		net = net.module
net.eval().to(device)

# def preprocess(image_path):
#     img = Image.open(image_path)

#     trf = T.Compose([T.Resize(450), 
#                     T.ToTensor(), 
#                     T.Normalize(mean = [0.485, 0.456, 0.406], 
#                                 std = [0.229, 0.224, 0.225])])

#     inp = trf(img).unsqueeze(0).cpu()

#     return inp

def preprocess(image_path, in_size=572, out_size=388):
    img_as_img = Image.open(image_path)
    # img_as_img.show()
    # Convert the image into numpy array
    img_as_np = np.asarray(img_as_img)

    pad_size = int((in_size - out_size)/2)
    img_as_np = np.pad(img_as_np, pad_size, mode="symmetric")
    img_as_np = multi_cropping(img_as_np,
                                crop_size=in_size,
                                crop_num1=2, crop_num2=2)

    # Empty list that will be filled in with arrays converted to tensor
    processed_list = []

    for array in img_as_np:

        # SANITY CHECK: SEE THE PADDED AND CROPPED IMAGES
        # array_image = Image.fromarray(array)

        # Normalize the cropped arrays
        img_to_add = normalization2(array, max=1, min=0)
        # Convert normalized array into tensor
        processed_list.append(img_to_add)

    img_as_tensor = torch.Tensor(processed_list)
    #  return tensor of 4 cropped images
    #  top left, top right, bottom left, bottom right respectively.
    return img_as_tensor

inp = preprocess(filename)

# print(inp.size())
# print(inp.size()[1])
# exit(1)

stacked_img = torch.Tensor([])
for index in range(inp.size()[1]):
    inp_t = Variable(inp[:, index, :, :].unsqueeze(0).cpu())

    output = net(inp_t)
    output = torch.argmax(output, dim=1).float()
    stacked_img = torch.cat((stacked_img, output))

div_arr = division_array(388, 2, 2, 512, 512)
img_cont = image_concatenate(stacked_img.data.numpy(), 2, 2, 512, 512)
final_img = (img_cont*255/div_arr)
final_img = final_img.astype("uint8")
# print(inp.shape)

plt.imshow(final_img)
plt.axis('off')
plt.title(f"output")
plt.show()

# print(x.shape)
# del net
# del x
# print(x.shape)