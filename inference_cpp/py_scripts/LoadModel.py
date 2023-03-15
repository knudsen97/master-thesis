import torch
import segmentation_models_pytorch
import cv2 as cv
import sys
import numpy as np
import torchvision.transforms as tf

def LoadModel():
    path_to_model = "../models/first_model.pt"

    encoder_depth = 3
    # Get ResNet101 pretrained model to use as encoder
    model = segmentation_models_pytorch.Unet(encoder_name='resnet101',
                                            encoder_weights='imagenet',
                                            classes=3,
                                            activation=None,
                                            encoder_depth=encoder_depth,
                                            decoder_channels = (128, 64, 32),
                                            #  decoder_use_batchnorm = True,
                                            #  decoder_attention_type = "scse",
                                            #  encoder_depth=2,
                                            #  decoder_channels = (32, 16)
                                            )

    model.load_state_dict(torch.load(path_to_model))
    model.eval()
    return model

def Inference(image, model):
    # Convert image to RGB
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    # Inference
    org_imag = image

    # Define differnt constants
    input_size = (640, 480) # for opencv
    scaled_size = (128, 160) # for torchvision

    # Found using the bottom function
    mean =  [0.4543, 0.3444, 0.2966]#[0.4352, 0.3342, 0.2835] 
    std_dev = [0.2198, 0.2415, 0.2423]#[0.2291, 0.2290, 0.2181]


    transforms = tf.Compose([
        tf.ToTensor(), # This also converts from 0,255 to 0,1
        tf.Resize(scaled_size),
        tf.Normalize(mean, std_dev)
    ])

    with torch.no_grad():
        image = transforms(image)
        output_tensor = model(image.unsqueeze(0))
        # denormalize output tensor
        denormalized_output = output_tensor * torch.tensor(std_dev).view(3, 1, 1) + torch.tensor(mean).view(3, 1, 1)

        # convert output tensor to numpy array
        output_image = (denormalized_output.clamp(0, 1) * 255).to('cpu').squeeze().numpy().astype(np.uint8)
    
    # Transpose image to (H, W, C)
    output = output_image.transpose(1, 2, 0)

    # Convert image to RGB
    output = cv.cvtColor(output, cv.COLOR_BGR2RGB)

    # Resize output to original size
    output = cv.resize(output, input_size, interpolation=cv.INTER_LINEAR)

    
    return output


if __name__ == "__main__":
    idName = "000028-0"
    datapath = "../data/"
    sys.path.append(datapath)
    image = cv.imread(f"{datapath}color-input/{idName}.png")
    if image is None:
        print("Could not read the image.")
        sys.exit(1)
    model = LoadModel()
    output = Inference(image, model)
    cv.imshow("image", image)
    cv.imshow("output", output)
    cv.waitKey(0)
