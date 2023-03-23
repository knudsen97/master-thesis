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
    def __init__(self, data_dir, synthetic, transform=None, transform_augmentations=None):
        self.synthetic = synthetic
        self.data_dir = data_dir

        self.transform = transform
        self.transform_augmentations = transform_augmentations

        self.images = sorted(os.listdir(os.path.join(data_dir, 'color-input')))
        self.masks  = sorted(os.listdir(os.path.join(data_dir, 'label')))
        
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
        img_path    = os.path.join(self.data_dir, 'color-input', self.images[idx])
        mask_path   = os.path.join(self.data_dir, 'label', self.masks[idx])

        # Read image and mask
        image = cv2.imread(img_path)
        mask  = cv2.imread(mask_path)

        # Convert to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask  = cv2.cvtColor(mask,  cv2.COLOR_BGR2RGB)
        if self.synthetic:
            target_mask = mask
        else:
            target_mask = self.create_mask(mask)

        # Apply transformations
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
            target_mask = self.transform(target_mask)
        if self.transform_augmentations:
            image = self.transform_augmentations(image)

        # Permute the image dimensions to (H, W, C)
        image = image.permute(1, 2, 0)
        mask = mask.permute(1, 2, 0)
        target_mask = target_mask.permute(1, 2, 0)

        return image, target_mask, mask

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


class EarlyStopper():
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def load_model(depth, device):
  if depth==3:
    decoder_channels = (64, 32, 16)
  elif depth==4:
    decoder_channels = (128, 64, 32, 16)
  elif depth==5:
    decoder_channels = (256, 128, 64, 32, 16)
  else:
    print("Choose a depth between 3 and 5! Using default 3.")
    decoder_channels = (64, 32, 16)
    depth = 3

  # Get ResNet101 pretrained model to use as encoder
  model = segmentation_models_pytorch.Unet(encoder_name='resnet101', 
                                          encoder_weights='imagenet', 
                                          classes=3, 
                                          activation=None,
                                          encoder_depth=depth, 
                                          decoder_channels = decoder_channels,
                                          # decoder_use_batchnorm = True,
                                          # decoder_attention_type = "scse",
                                          )

  # print(model)
  model.encoder.train = False
  model.decoder.train = False
  model.segmentation_head.train = True
  # Move model to GPU
  model.to(device) #1GB GPU
  return model

def get_criterion():
  criterion = nn.CrossEntropyLoss()
  return criterion

def get_optimizer(model, lr=1e-04, l2_penal=0.0, optimizer_name="Adam"):
  # Define optimizer
  if optimizer_name == "Adam":
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_penal)
  elif optimizer_name == "Adamax":
    optimizer = torch.optim.Adamax(model.parameters(), lr=lr, weight_decay=l2_penal)
  return optimizer

def f1_score(recall, precision):
    return 2 * (recall * precision) / (recall + precision)

def main():
    # Found using the bottom function
    mean =  [0.4543, 0.3444, 0.2966]#[0.4352, 0.3342, 0.2835] 
    std = [0.2198, 0.2415, 0.2423]#[0.2291, 0.2290, 0.2181]

    # input_size = (480, 640)
    scaled_size = (128, 160)
    transforms = tf.Compose([
        tf.ToTensor(), # This also converts from 0,255 to 0,1
        tf.Resize(scaled_size),
    ])

    augmentations = "  Resize((128,160))\n  ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)\n   GaussianBlur(3)"
    transform_augmentations = tf.Compose([
        tf.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        tf.GaussianBlur(3),
        tf.Normalize(mean, std),
    ])


    # Create dataset
    data_dir = 'data'
    data_syn_dir = 'data_synthetic'
    dataset_real = SegNetDataset(data_dir=data_dir, synthetic=False, transform=transforms, transform_augmentations=transform_augmentations)
    dataset_syn = SegNetDataset(data_dir=data_syn_dir, synthetic=True, transform=transforms, transform_augmentations=transform_augmentations)

    # Split dataset into train and test
    real_train_data, real_test_data = data.random_split(dataset_real, [int(len(dataset_real)*0.8), len(dataset_real)-int(len(dataset_real)*0.8)])
    synthetic_train_data, synthetic_test_data = data.random_split(dataset_syn, [int(len(dataset_syn)*0.8), len(dataset_syn)-int(len(dataset_syn)*0.8)])

    # Concatenate the datasets
    train_data = data.ConcatDataset([real_train_data, synthetic_train_data])
    test_data  = data.ConcatDataset([real_test_data, synthetic_test_data])
    
    # Find GPU device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device)
    assert device == torch.device("cuda:0") 

    # Define the hyperparameters you want to search over
    learning_rates = [1e-3, 1e-5, 1e-7]
    depths = [3, 4, 5]
    batch_sizes = [2, 4, 8]
    epochs = 1
    results = {}


    train_id = 0
    for lr in learning_rates:
        for depth in depths:
            for bs in batch_sizes:
                # Create dataloaders
                train_loader = data.DataLoader(train_data, batch_size=bs, shuffle=True, num_workers=4)
                test_loader  = data.DataLoader(test_data, batch_size=bs, shuffle=False, num_workers=4)
                criterion = get_criterion()
                model = load_model(depth, device)
                optimizer = get_optimizer(model, lr=lr, l2_penal=0.0, optimizer_name="Adam")
                early_stopper = EarlyStopper(patience=4, min_delta=0.05)

                train_losses, test_losses     = [], []
                train_recall, train_precision = [], []
                test_recall, test_precision   = [], []

                csv_file_path = f'results_grid_search/results{train_id}.csv'
                with open(csv_file_path, 'w', newline='') as csv_file:
                    writer = csv.writer(csv_file)

                    # Write the headers if the file is empty
                    if csv_file.tell() == 0:
                        # Write the hyperparameters used for this test
                        writer.writerow(['epochs', 'encoder_depth', 'lr', 'batch_size', 'l2_penalization'])
                        writer.writerow([epochs, depth, lr, bs, 0.0])

                        # Write the results headers
                        writer.writerow(['train_losses', 'test_losses', 'train_recall', 'train_precision', 'test_recall', 'test_precision'])

                    # Start training and testing
                    for epoch in range(epochs):
                        # Print current epoch, lr, depth, batch size
                        print(f"Epoch: {epoch}, lr: {lr}, depth: {depth}, batch size: {bs}")
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
                        if early_stopper.early_stop(test_losses[-1]):
                            break
                results[(lr, depth, bs)] = (test_recall[-1], test_precision[-1], f1_score(test_recall[-1], test_precision[-1]))
                train_id += 1


    # Print the results to a csv file
    with open('results_grid_search/grid_search_results.csv', 'w') as f:
        f.write("learning_rate,depth,batch_size,recall,precision,f1_score\n")
        for key in results.keys():
            lr, depth, bs = key

            # Extract the values with 2 decimal places
            recall = round(results[key][0], 2)
            precision = round(results[key][1], 2)
            f1 = round(results[key][2], 2)

            f.write(f"{lr},{depth},{bs},{recall},{precision},{f1}\n")
 




if __name__ == "__main__":
    main()