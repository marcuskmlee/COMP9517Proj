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

dataset = "DIC-C2DH-HeLa"
sequences = ["Sequence_2", "Sequence_3", "Sequence_4"]
masks = ["Sequence_2_Preds", "Sequence_3_Preds", "Sequence_4_Preds"]
folders = zip(sequences, masks)

for image_folder, mask_folder in folders:
        print(f"dataset: ../../Data/{dataset}/{image_folder}/")
        SEM_test = SEMDataTest(f"../../Data/{dataset}/{image_folder}/")

        SEM_test_load = \
        torch.utils.data.DataLoader(dataset=SEM_test,
                                        num_workers=3, batch_size=1, shuffle=False)

        model_path = "./model_epoch_2290.pwf"

        net = torch.load(model_path, map_location=torch.device('cpu'))
        if isinstance(net,torch.nn.DataParallel):
                net = net.module
        net.eval().to("cpu")

        print(f"run on: ../../Data/{dataset}/{mask_folder}/")
        test_model(net,
                SEM_test_load, 2290, f"../../Data/{dataset}/{mask_folder}/")