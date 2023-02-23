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


def inference(image: np.ndarray):
    model = torch.load('py_scripts/model.pth')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = tf.ToTensor()(image)

    output = model(image.unsqueeze(0).to(device)).squeeze().cpu().detach().numpy().astype('float32')
    return output

if __name__ == '__main__':
    image = cv2.imread('../_data/color-input/000000-0.png')
    output = inference(image)
    plt.imshow(output)
    plt.show()
