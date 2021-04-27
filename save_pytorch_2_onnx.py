import os
import random
import torch
from torch.autograd import Variable
import torchvision.transforms as standard_transforms
import misc.transforms as own_transforms
import pandas as pd

from models.CC import CrowdCounter
from config import cfg
from misc.utils import *
import scipy.io as sio
from PIL import Image, ImageOps
import sys

#torch.cuda.set_device(0)
torch.backends.cudnn.benchmark = True

save_path = '/home/stone/ai/mldata/crowd_density_counting/pb/cdc.onnx'
model_path = '/home/stone/ai/mldata/crowd_density_counting/ckpt/05-ResNet-50_all_ep_35_mae_32.4_mse_76.1.pth'


def main():

    net = CrowdCounter([], cfg.NET)
    net.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))
    #net.cuda()
    net.eval()

    x = Variable(torch.randn(1, 3, 768, 1024), requires_grad=True).cpu()
    x_1 = Variable(torch.randn(1, 768, 1024), requires_grad=True).cpu()

    torch_out = torch.onnx._export(net, # model being run
        (x, x_1),                      # model input (or a tuple for multiple inputs)
        save_path, # where to save the model (can be a file or file-like object)
        export_params=True,
        #input_names=["input"],output_names=["output"],dynamic_axes={"input":{0:"batch_size"},"output":{0:"batch_size"}})      # store the trained parameter weights inside the model file
        input_names=["input"],output_names=["output"])      # store the trained parameter weights inside the model file
    print('torch_out={}'.format(torch_out))


if __name__ == '__main__':
    main()
