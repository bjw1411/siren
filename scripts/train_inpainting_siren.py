# Originally forked from https://github.com/dalmia/siren
# Authors: Aman Dalmia, Ben Williamson
# Trains SIREN model for inpainting task on a single image. Implements learning rate decay.
import os
from datetime import datetime, timedelta
from PIL import Image
import logging
import numpy as np
import matplotlib.pyplot as plt
import csv
from tqdm import tqdm
import torch
from torch.utils.data import TensorDataset, DataLoader
from siren import SIREN
from utils import set_logger

# Model Hyper parameters
# For the experiment we are keeping these constant
SAMPLING_RATIO = 0.1
BATCH_SIZE = 8192
EPOCHS = 5000

# Initial learning rate
LEARNING_RATE = 0.0015
# Learning Rate decay from exponential
decay_rate = 0.9997

# Image to train on
filename = 'bread.png'

# Open the image with torch
img_filepath = 'data/' + filename
img_raw = np.array(Image.open(img_filepath))
img_ground_truth = torch.from_numpy(img_raw).float()

# Get values from the image
rows, cols, channels = img_ground_truth.shape
pixel_count = rows * cols

# With sampling rate find the amount of sampled pixels
sampled_pixel_count = int(pixel_count * SAMPLING_RATIO)

# build tensors
def build_train_tensors():
    img_mask_x = torch.from_numpy(np.random.randint(0, rows, sampled_pixel_count))
    img_mask_y = torch.from_numpy(np.random.randint(0, cols, sampled_pixel_count))

    img_train = img_ground_truth[img_mask_x, img_mask_y]

    img_mask_x = img_mask_x.float() / rows
    img_mask_y = img_mask_y.float() / cols

    img_mask = torch.stack([img_mask_x, img_mask_y], dim=-1)

    return img_mask, img_train


img_mask, img_train = build_train_tensors()

train_dataset = TensorDataset(img_mask, img_train)
train_dataloader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

# Using SIREN build the model
# Model Parameters
layers = [256, 256, 256, 256, 256]
in_features = 2
out_features = 4
# Initialize with default siren intializer
initializer = 'siren'
w0 = 1.0
w0_initial = 30.0
c = 6
model = SIREN(
    layers, in_features, out_features, w0, w0_initial,
    initializer=initializer, c=c)

model.train()

BATCH_SIZE = min(BATCH_SIZE, len(img_mask))
num_steps = int(len(img_mask) * EPOCHS / BATCH_SIZE)
print("Total training steps : ", num_steps)

# If CUDA is supported use it
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create the optimizer with the model parameters, and initial learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Apply a learning rate scheduler with exponential decay to the optimizer
learning_rate = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decay_rate)

criterion = torch.nn.MSELoss()

# Set directory to save model
checkpoint_dir = 'checkpoints/siren/inpainting/'
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

# Create logger
timestamp = datetime.now().strftime('%Y-%m-%d %H-%M-%S')
logdir = os.path.join('logs/siren/inpainting/', timestamp)
if not os.path.exists(logdir):
    os.makedirs(logdir)
set_logger(os.path.join(logdir, 'train.log'))

# Initialize lists used for measuring loss improvements
epoch_losses = []
epoch_improvements = []

# Initialize starting loss to infinite
best_loss = np.inf

# Save initial timestamp before training
timestamp = datetime.now()

# Loop over image
for epoch in range(EPOCHS):
    iterator = tqdm(train_dataloader, dynamic_ncols=True)

    losses = []
    
    # Train for each batch
    for batch in iterator:
        inputs, targets = batch
        predictions = model(inputs)
        loss = criterion(predictions, targets)
        losses.append(loss.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        iterator.set_description(
            "Epoch: {} | Loss {:.4f}".format(epoch, loss), refresh=True)

    avg_loss = torch.mean(torch.cat(losses)).item()
    epoch_losses.append([epoch, avg_loss])

    # At the end of the epoch decay the learning rate
    learning_rate.step()

    # If improved loss, save the epoch number and loss
    if avg_loss < best_loss:
        logging.info('Loss improved from {:.4f} to {:.4f}'.format(best_loss, avg_loss))

        best_loss = avg_loss
        epoch_improvements.append([epoch, best_loss])

        # Save the improved model
        torch.save({'network': model.state_dict()}, os.path.join(checkpoint_dir + 'model'))

# Save and print the total time model took to train
complete = datetime.now()
dif = complete - timestamp
before_str = before.strftime("%H:%M:%S")
logging.info('Started at: ' + before_str)
complete_str = complete.strftime("%H:%M:%S")
logging.info('Completed at: ' + complete_str)
logging.info('Total time: ' + str(dif.seconds//3600) + ':' + str((dif.seconds//60) % 60) + ":" + str(dif.seconds%60))

# Save a copy of the model to the saved models dir
torch.save(model.state_dict(), "saved models/"+filename+" model.pt")

# Write the epoch losses to csv files
with open('metrics/'+filename+' - avg losses.csv','w+') as losses_file:
    writer = csv.writer(losses_file)
    writer.writerows(epoch_losses)

with open('metrics/'+filename+' - improvements.csv','w+') as improvements_file:
    writer = csv.writer(improvements_file)
    writer.writerows(epoch_improvements)

# Save and plot the epoch losses
epoch_losses = np.array(epoch_losses)
plt.figure(0)
x,y = epoch_losses.T
plt.plot(x,y)
plt.title("Epoch vs Avg. Loss")
plt.xlabel("Epoch")
plt.ylabel("Avg. Loss")
plt.savefig('metrics/'+(filename.split('.')[0])+' - plot of losses')

epoch_improvements= np.array(epoch_improvements)
plt.figure(1)
x,y = epoch_improvements.T
plt.plot(x,y)
plt.title("Epoch vs Avg. Loss")
plt.xlabel("Epoch")
plt.ylabel("Avg. Loss")
plt.savefig('metrics/'+(filename.split('.')[0])+' - plot of improvement')




