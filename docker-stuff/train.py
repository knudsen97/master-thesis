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
    def __init__(self, data_dir, synthetic, transform=None, synthetic_data_count = 1000):#, transform_augmentations=None):
        self.synthetic = synthetic
        self.data_dir = data_dir
        self.synthetic_data_count = synthetic_data_count

        self.transform = transform
        # self.transform_augmentations = transform_augmentations

        self.images = sorted(os.listdir(os.path.join(data_dir, 'color-input')))
        self.masks  = sorted(os.listdir(os.path.join(data_dir, 'label')))

        if self.synthetic:
            self.images = self.images[:self.synthetic_data_count]
            self.masks = self.masks[:self.synthetic_data_count]
        
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
        # if self.transform_augmentations:
        #     image = self.transform_augmentations(image)

        # # Permute the image dimensions to (H, W, C)
        # image = image.permute(1, 2, 0)
        # mask = mask.permute(1, 2, 0)
        # target_mask = target_mask.permute(1, 2, 0)

        return image, target_mask, mask

class MapDataset(Dataset):
    """
    Given a dataset, creates a dataset which applies a mapping function
    to its items (lazily, only when an item is called).

    Note that data is not cloned/copied from the initial dataset.
    """
    def __init__(self, dataset, augmentations):
        self.dataset = dataset
        self.augmentations = augmentations

    def __getitem__(self, index):
        # print(self.dataset[index])
        image, target_mask, mask = self.dataset[index]
        if self.augmentations is not None:
          image = self.augmentations(image)

        # Permute the image dimensions to (H, W, C)
        image = image.permute(1, 2, 0)
        mask = mask.permute(1, 2, 0)
        target_mask = target_mask.permute(1, 2, 0)
        return image, target_mask, mask

    def __len__(self):
        return len(self.dataset)


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
    epochs_default = 50
    encoder_depth_default = 3
    lr_default = 1e-04
    batch_size_default = 4
    l2_penalization_default = 0.0
    decoder_use_batchnorm = False
    decoder_attention_type = None
    optimizer_type_default = "Adam"
    id_default = '0'

    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-epochs', type=int, default=epochs_default, help='Number of epochs')
    parser.add_argument('-encoder_depth', type=int, default=encoder_depth_default, help='Depth of encoder')
    parser.add_argument('-lr', type=float, default=lr_default, help='Learning rate')
    parser.add_argument('-batch_size', type=int, default=batch_size_default, help='Batch size')
    parser.add_argument('-l2', type=float, default=l2_penalization_default, help='L2 penalization (weight decay)')
    parser.add_argument('-decoder_use_batchnorm', type=bool, default=decoder_use_batchnorm, help='Use batchnorm in decoder')
    parser.add_argument('-decoder_attention_type', type=str, default=decoder_attention_type, help='Attention type in decoder')
    parser.add_argument('-optimizer', type=str, default=optimizer_type_default, help='Type of optimizer: Adam, Adamax')
    parser.add_argument('-id', type=str, default=id_default, help='id used for saving the results')


    return parser.parse_args()

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

def load_model(depth, device, decoder_use_batchnorm, decoder_attention_type):
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
                                            decoder_use_batchnorm = decoder_use_batchnorm,
                                            decoder_attention_type = decoder_attention_type,
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

def main():
    args = arg_parser()

    print(f"Using the following hyperparameters: {args}")
   
    # ImageNet mean and std: (Found using the bottom function)
    mean =  [0.485, 0.456, 0.406] #[0.4543, 0.3444, 0.2966]#[0.4352, 0.3342, 0.2835] 
    std = [0.229, 0.224, 0.225]#[0.2198, 0.2415, 0.2423]#[0.2291, 0.2290, 0.2181]
    scaled_size = (128, 160)
    
    # Transforms to be applied to all images, masks and target masks
    transforms = tf.Compose([
        tf.ToTensor(), # This also converts from 0,255 to 0,1
        tf.Resize(scaled_size),
    ])

    # Augmentations for train and test images
    transform_augmentations_train = tf.Compose([
        tf.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        tf.GaussianBlur(3),
        tf.Normalize(mean, std)
    ])
    transform_augmentations_test = tf.Compose([
        tf.GaussianBlur(3),
        tf.Normalize(mean, std)
    ])


    # Create dataset
    data_dir = 'data'
    data_syn_dir = 'data_synthetic'
    dataset_real = SegNetDataset(data_dir=data_dir, synthetic=False, transform=transforms)#, transform_augmentations=transform_augmentations)
    dataset_syn = SegNetDataset(data_dir=data_syn_dir, synthetic=True, transform=transforms)#, transform_augmentations=transform_augmentations)

    # Split dataset into train and test
    real_train_data, real_test_data = data.random_split(dataset_real, [int(len(dataset_real)*0.8), len(dataset_real)-int(len(dataset_real)*0.8)])
    synthetic_train_data, synthetic_test_data = data.random_split(dataset_syn, [int(len(dataset_syn)*0.8), len(dataset_syn)-int(len(dataset_syn)*0.8)])

    # Concatenate the datasets
    train_data = data.ConcatDataset([real_train_data, synthetic_train_data])
    test_data  = data.ConcatDataset([real_test_data, synthetic_test_data])

    # Create new datasets with augmentations for training and nothing for test/validation.
    train_data = MapDataset(train_data, augmentations=transform_augmentations_train)
    test_data = MapDataset(test_data, augmentations=transform_augmentations_test)

    # Create dataloaders
    batch_size = args.batch_size
    train_loader = data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader  = data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4)

    # Load a test batch
    images, target_mask, masks = next(iter(train_loader))
    print(f"shapes: {images.shape}, {target_mask.shape}, {masks.shape}")

    # Find GPU device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("Using device: ", device)
    assert device == torch.device("cuda:0") 

    # Get parameters for model and load model
    encoder_depth = args.encoder_depth
    decoder_use_batchnorm = args.decoder_use_batchnorm
    decoder_attention_type = args.decoder_attention_type
    model = load_model(depth=encoder_depth, device=device, decoder_use_batchnorm=decoder_use_batchnorm, decoder_attention_type=decoder_attention_type)

    print("batchnorm: ", decoder_use_batchnorm)


    # Define optimizer
    criterion = get_criterion()
    # criterion = nn.CrossEntropyLoss()

    # Define optimizer
    l2_penalization = args.l2
    lr = args.lr
    optimizer_type = args.optimizer
    optimizer = get_optimizer(model, lr, l2_penalization, optimizer_type)
    # optimizer_name = "Adamax"
    # optimizer = torch.optim.Adamax(model.parameters(), lr=lr, weight_decay=l2_penalization)
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_penalization)


    # Train model
    epochs = args.epochs
    train_losses, test_losses     = [], []
    train_recall, train_precision = [], []
    test_recall, test_precision   = [], []


    # Early stopper
    early_stopper = EarlyStopper(patience=5, min_delta=0.01)


    # Define the file path and open the CSV file in append mode and save the results in results folder
    csv_file_path = f'results/results_{args.id}.csv'
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
            
            if early_stopper.early_stop(test_losses[-1]):
                break
    
    
    # Save the model
    torch.save(model.state_dict(), f'models/unet_resnet101_{args.id}.pt')

    # Save model to load in c++
    try:
        torch.jit.save(torch.jit.script(model), f'jit_models/unet_resnet101_{args.id}.pt')
    except:
        print("Could not save model with jit.")



if __name__ == "__main__":
    main()