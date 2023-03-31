import os
import numpy as np
import torchvision.models.segmentation
import torch
import torchvision.transforms as tf
from torch.utils.data import Dataset
import cv2
from torch.utils import data
import matplotlib.pyplot as plt

import segmentation_models_pytorch

def test():
    # model = segmentation_models_pytorch.Unet(encoder_name='resnet101', 
    #                                         encoder_weights='imagenet', 
    #                                         classes=3, 
    #                                         activation=None,
    #                                         encoder_depth=3, 
    #                                         decoder_channels = (128, 64, 32),
    #                                         )
    # model = torch.load('models/first_model.pt')
    model = torch.jit.load('models/temp_model.pt')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print(f"cuda availability: {torch.cuda.is_available()}")
    model
    input = torch.ones(1, 3, 128, 160)
    output = model(input)
    output = output.squeeze().cpu().detach()
    output = output.permute(1, 2, 0)
    output = output.numpy().astype('float32')

    # print first 3x3 pixel values
    import sys
    np.set_printoptions(threshold=sys.maxsize)
    np.set_printoptions(suppress = True)
    print(output)


    cv2.imshow('output', output)
    cv2.waitKey(0)


def inference(image: np.ndarray):
    model = torch.load('models/first_model.pt')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = tf.ToTensor()(image)

    output = model(image.unsqueeze(0).to(device)).squeeze().cpu().detach().numpy().astype('float32')
    return output

if __name__ == '__main__':
    # image = cv2.imread('../_data/color-input/000000-0.png')
    # output = inference(image)
    # plt.imshow(output)
    # plt.show()
    test()
