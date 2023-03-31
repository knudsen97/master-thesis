import torch
import segmentation_models_pytorch
import cv2 as cv
import sys
import numpy as np
import torchvision.transforms as tf
from PIL import Image

def LoadModel():
    print("LoadModel")
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    path_to_model = "../../models/first_model.pt"
    path_with_no_model = "../../models/"
    sys.path.append(path_with_no_model)

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
    # model.to(device)
    return model

def Inference(image, model):
    # Convert image to RGB
    # image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    # image = Image.fromarray(image)

    # Inference
    # org_imag = image

    # Define differnt constants
    scaled_size = (128, 160) # for torchvision

    # Found using the bottom function
    mean =  [0.4543, 0.3444, 0.2966]#[0.4352, 0.3342, 0.2835] 
    std_dev = [0.2198, 0.2415, 0.2423]#[0.2291, 0.2290, 0.2181]

    print("making b 128x160x3")
    b = np.ndarray((128, 160, 3), dtype=np.uint8)
    # fill b with 1s
    b.fill(1)

    print(f"b shape: {b.shape}")

    print("making a")
    a = tf.ToTensor()(b)
    print("a was made")

    print(f"image shape: {image.shape}")
    print(f"b dtype: {b.dtype}")
    print(f"image dtype: {image.dtype}")
    print("image to tensor")
    # tf.ToTensor()(image)
    transforms = tf.Compose([
        tf.ToTensor(), # This also converts from 0,255 to 0,1
        tf.Resize(scaled_size),
        tf.Normalize(mean, std_dev)
    ])
    print(f"-------------b shape: {b.shape}")
    b = tf.ToTensor()(b)
    print(f"-------------b shape: {b.shape}")
    # image = transforms(b)

    debug_image = image.numpy().transpose(1, 2, 0)
    cv.imshow("debug_image", debug_image)
    print("Inference")
    with torch.no_grad():
        print("transform image")

        print("Running model")
        output_tensor = model(image.unsqueeze(0))
        # denormalize output tensor
        print("Denormalizing output")
        denormalized_output = output_tensor * torch.tensor(std_dev).view(3, 1, 1) + torch.tensor(mean).view(3, 1, 1)

        # convert output tensor to numpy array
        print("Converting output to numpy array")
        output_image = (denormalized_output.clamp(0, 1) * 255).to('cpu').squeeze().numpy().astype(np.uint8)
    print("Inference done")
    
    # Transpose image to (H, W, C)
    output = output_image.transpose(1, 2, 0)

    # Convert image to RGB
    output = cv.cvtColor(output, cv.COLOR_BGR2RGB)

    # Resize output to original size
    input_size = (640, 480) # for opencv
    output = cv.resize(output, input_size, interpolation=cv.INTER_LINEAR)

    print("Returning output")
    return output


if __name__ == "__main__":
    idName = "000000-0"
    datapath = "../../data/"
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
