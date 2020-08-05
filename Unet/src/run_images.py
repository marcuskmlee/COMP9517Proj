from advanced_model import CleanU_Net
from dataset import *
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from PIL import Image
from modules import *
from save_history import *
import matplotlib.pyplot as plt

dataset = ["DIC-C2DH-HeLa", "Fluo-N2DL-HeLa", "PhC-C2DL-PSC"]

SEM_test = SEMDataTest(f"../../../Data/{dataset[0]}/Sequence 1/")

SEM_test_load = \
    torch.utils.data.DataLoader(dataset=SEM_test,
                                num_workers=3, batch_size=1, shuffle=False)

model_path = "../../model_epoch_2290.pwf"

net = torch.load(model_path, map_location=torch.device('cpu'))
if isinstance(net,torch.nn.DataParallel):
		net = net.module
net.eval().to("cpu")


test_model(net,
           SEM_test_load, 2290, f"../../../Data/{dataset[0]}/Sequence_1_Preds/")

# img, name = test_SEM(net, 
#     SEM_test_load, f"../../../Data/{dataset[0]}/Sequence_1_Preds/")

# plt.imshow(img)
# plt.axis('off')
# plt.title(name)
# plt.show()