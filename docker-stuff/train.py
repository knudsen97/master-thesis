import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils import data
import torchvision.transforms as tf
import segmentation_models_pytorch
import cv2

from tqdm import tqdm
import csv
import argparse

# Create custom dataset class
class SegNetDataset(Dataset):
    def __init__(self, root_dir, transform=None, transform_augmentations=None):
        self.root_dir = root_dir
        self.transform = transform
        self.transform_augmentations = transform_augmentations

        # self.toTensor = tf.Compose([tf.ToTensor()])
        self.images = os.listdir(os.path.join(root_dir, 'color-input'))
        # self.depths = os.listdir(os.path.join(root_dir, 'depth-input'))
        self.masks  = os.listdir(os.path.join(root_dir, 'label'))
        
    def __len__(self):
        return len(self.images)

    def create_mask(self, mask):
        # Create empty target mask
        target_mask = np.array(mask, dtype=np.uint8)

        # Create mask for each class
        target_mask[np.where((mask == [0, 0, 0]).all(axis=2))]       = [255, 0, 0]
        target_mask[np.where((mask == [128, 128, 128]).all(axis=2))] = [0, 255, 0]
        target_mask[np.where((mask == [255, 255, 255]).all(axis=2))] = [0, 0, 255]

        return target_mask

    def __getitem__(self, idx):
        img_path    = os.path.join(self.root_dir, 'color-input', self.images[idx])
        mask_path   = os.path.join(self.root_dir, 'label', self.masks[idx])
        
        # Read image and mask
        image = cv2.imread(img_path)
        mask  = cv2.imread(mask_path)

        # Convert to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask  = cv2.cvtColor(mask,  cv2.COLOR_BGR2RGB)
        target_mask = self.create_mask(mask)



        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
            target_mask = self.transform(target_mask)        
        
        if self.transform_augmentations:
            image = self.transform_augmentations(image)

        # print(f"image/mask/target sizes: {image.shape} / {mask.shape} / {target_mask.shape}")
        # Permute the image dimensions to (H, W, C)
        image = image.permute(1, 2, 0)
        mask = mask.permute(1, 2, 0)
        target_mask = target_mask.permute(1, 2, 0)

        return image, target_mask, mask    #, depth

# Define function to calculate accuracy
def accuracy(outputs, targets):
    # Permute to (B, H, W, C)
    outputs = outputs.permute(0, 2, 3, 1)
    targets = targets.permute(0, 2, 3, 1)

    # Find max value and index of max value
    max_values, _ = torch.max(outputs, dim=3) # Here dim=3 because we want to find max value for class
    outputs_targets = (outputs == max_values.unsqueeze(3)).float().round()

    # Find true positives, false positives, false negatives
    true_positives  = torch.where((outputs_targets == targets) & (targets == 1), 1, 0).sum(dim=(0, 1, 2))
    false_positives = torch.where((outputs_targets != targets) & (targets == 0), 1, 0).sum(dim=(0, 1, 2))
    false_negatives = torch.where((outputs_targets != targets) & (targets == 1), 1, 0).sum(dim=(0, 1, 2))

    # How to deal with 0 values
    true_positives  = torch.where(true_positives  == 0, 1, true_positives)

    # Calculate recall and precision
    recall = true_positives / (true_positives + false_negatives)
    precision = true_positives / (true_positives + false_positives)

    return recall.mean(), precision.mean()

# Define train function that returns loss for each epoch and the accuracy
def train(model, train_loader, criterion, optimizer, device):
    train_loss = 0.0
    train_recall_running, train_precision_running = 0.0, 0.0

    for images, target_mask, masks in tqdm(train_loader):
        images = images.to(device)
        target_mask = target_mask.to(device)

        images = images.permute(0, 3, 1, 2)
        target_mask = target_mask.permute(0, 3, 1, 2)

        # print(images.shape)
        optimizer.zero_grad()
        outputs = model(images)

        # Calculate accuracy
        train_recall, train_precision = accuracy(outputs, target_mask)
        train_recall_running += train_recall
        train_precision_running += train_precision

        # Calculate loss
        loss = criterion(outputs, target_mask.argmax(dim=1))
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    return train_loss/len(train_loader), train_recall_running/len(train_loader), train_precision_running/len(train_loader)

# Define test function
def test(model, test_loader, criterion, device):
    # model.eval()
    test_loss = 0.0
    test_recall_running, test_precision_running = 0.0, 0.0

    with torch.no_grad():
        for images, target_mask, masks in tqdm(test_loader):
            images = images.to(device)
            target_mask = target_mask.to(device)
            
            images = images.permute(0, 3, 1, 2)
            target_mask = target_mask.permute(0, 3, 1, 2)

            outputs = model(images)
           
            # Calculate loss
            loss = criterion(outputs, target_mask.argmax(dim=1))
            test_loss += loss.item()

            # Calculate accuracy
            test_recall, test_precision = accuracy(outputs, target_mask)
            test_recall_running += test_recall
            test_precision_running += test_precision

    return test_loss/len(test_loader), test_recall_running/len(test_loader), test_precision_running/len(test_loader)

def arg_parser():
    # Default values for command-line arguments
    epochs_default = 30
    encoder_depth_default = 3
    lr_default = 0.00001
    batch_size_default = 4
    l2_penalization_default = 0.01
    id_default = 0
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-epochs', type=int, default=epochs_default, help='Number of epochs')
    parser.add_argument('-encoder_depth', type=int, default=encoder_depth_default, help='Depth of encoder')
    parser.add_argument('-lr', type=float, default=lr_default, help='Learning rate')
    parser.add_argument('-batch_size', type=float, default=batch_size_default, help='Batch size')
    parser.add_argument('-l2', type=float, default=l2_penalization_default, help='L2 penalization (weight decay)')
    parser.add_argument('-id', type=int, default=id_default, help='id used for saving the results')

    return parser.parse_args()

def main():
    args = arg_parser()

    print(f"Using the following hyperparameters: {args}")

    # input_size = (480, 640)
    scaled_size = (128, 160)
    transforms = tf.Compose([
        tf.ToTensor(), # This also converts from 0,255 to 0,1
        tf.Resize(scaled_size),
    ])

    augmentations = "  Resize((128,160))\n  ColorJitter((0.7,1), (1), (0.7,1.3), (-0.1,0.1))\n   GaussianBlur(3)"
    transform_augmentations = tf.Compose([
        tf.ColorJitter((0.7,1), (1), (0.7,1.3), (-0.1,0.1)),
        tf.GaussianBlur(3),
    ])


    dataset = SegNetDataset(root_dir='data', transform=transforms, transform_augmentations=transform_augmentations)

    # Split dataset into train and test
    train_data, test_data = data.random_split(dataset, [int(len(dataset)*0.8), len(dataset)-int(len(dataset)*0.8)])

    batch_size = args.batch_size
    # Create dataloaders
    train_loader = data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader  = data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4)

    # Load a test batch
    images, target_mask, masks = next(iter(train_loader))
    print(f"shapes: {images.shape}, {target_mask.shape}, {masks.shape}")

    # Find GPU device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("Using device: ", device)
    assert device == torch.device("cuda:0") 

    encoder_depth = args.encoder_depth
    decoder_channels = "(256, 128, 64)"

    # Get ResNet101 pretrained model to use as encoder
    model = segmentation_models_pytorch.Unet(encoder_name='resnet101', 
                                            encoder_weights='imagenet', 
                                            classes=3, 
                                            activation=None,
                                            encoder_depth=encoder_depth, 
                                            decoder_channels = (256, 128, 64),
                                            decoder_use_batchnorm = True,
                                            decoder_attention_type = "scse",
                                            )

    # print(model)
    model.encoder.train = False
    model.decoder.train = True
    model.segmentation_head.train = True

    # Move model to GPU
    model.to(device) #1GB GPU

    # Define optimizer
    criterion = nn.CrossEntropyLoss()
    l2_penalization = args.l2
    lr = args.lr
    # optimizer_name = "Adamax"
    optimizer = torch.optim.Adamax(model.parameters(), lr=lr, weight_decay=l2_penalization)
    

    # Train model
    epochs = args.epochs
    train_losses, test_losses     = [], []
    train_recall, train_precision = [], []
    test_recall, test_precision   = [], []

    # Define the file path and open the CSV file in append mode and save the results in results folder
    csv_file_path = f'results/results{args.id}.csv'
    with open(csv_file_path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)

        # Write the headers if the file is empty
        if csv_file.tell() == 0:
            # Write the hyperparameters used for this test
            writer.writerow(['epochs', 'encoder_depth', 'lr', 'batch_size', 'l2_penalization'])
            writer.writerow([epochs, encoder_depth, lr, batch_size, l2_penalization])

            # Write the results headers
            writer.writerow(['train_losses', 'test_losses', 'train_recall', 'train_precision', 'test_recall', 'test_precision'])

        # Start training and testing
        for epoch in range(epochs):
            # Train
            train_loss, train_rec, train_prec = train(model, train_loader, criterion, optimizer, device)
            train_losses.append(train_loss)
            train_recall.append(train_rec)
            train_precision.append(train_prec)

            # Print
            print(f'Epoch {epoch}, train loss: {train_losses[-1]:.4f}, train recall/precision: {train_recall[-1]:.4f}/{train_precision[-1]:.4f}')

            # Test
            test_loss, test_rec, test_prec = test(model, test_loader, criterion, device)
            test_losses.append(test_loss)
            test_recall.append(test_rec)
            test_precision.append(test_prec)   

            # Print
            print(f'Epoch {epoch}, test loss: {test_losses[-1]:.4f}, test recall/precision: {test_recall[-1]:.4f}/{test_precision[-1]:.4f}')
            writer.writerow([train_losses[-1], test_losses[-1], train_recall[-1].item(), train_precision[-1].item(), test_recall[-1].item(), test_precision[-1].item()])




if __name__ == "__main__":
    main()