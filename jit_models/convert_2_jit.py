#!/usr/bin/env python3
import numpy
import torch
import os
import argparse
import segmentation_models_pytorch



def main():
    # Read arguments
    args = argparse.ArgumentParser()
    args.add_argument('-f', '--file_name', type=str, default=None, help='name of the file')
    args = args.parse_args()
    if args.file_name is None:
        raise Exception('Please provide path to model')

    # Load model
    model = segmentation_models_pytorch.Unet(encoder_name='resnet101', 
                                            encoder_weights='imagenet', 
                                            classes=3, 
                                            activation=None,
                                            encoder_depth=3, 
                                            decoder_channels = (128, 64, 32), # 256, 128, 64, 32, 16
                                            decoder_use_batchnorm=True,
                                            decoder_attention_type='scse')
    model_name = args.file_name
    file_path = os.path.realpath(__file__)
    model_path = os.path.join(os.path.dirname(file_path), model_name)
    print(model_path)
    model.load_state_dict(torch.load(model_path))
    model.eval()


    # Generate jit model
    x = torch.randn(1, 3, 128, 160)
    traced_script_module = torch.jit.trace(model, x)
    new_model_name = model_path.split('.')[0] + '_jit.pt'
    traced_script_module.save(new_model_name)

if __name__ == '__main__':
    main()