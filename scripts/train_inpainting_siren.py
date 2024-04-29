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

SAMPLING_RATIO = 0.1
#original 8192
BATCH_SIZE = 8192
EPOCHS = 500
LEARNING_RATE = 0.0005

filename = 'celtic_spiral_knot.jpg'

# Image Reference - http://earthsongtiles.com/celtic_tiles.html
img_filepath = 'data/' + filename
img_raw = np.array(Image.open(img_filepath))
img_ground_truth = torch.from_numpy(img_raw).float()

rows, cols, channels = img_ground_truth.shape
pixel_count = rows * cols
sampled_pixel_count = int(pixel_count * SAMPLING_RATIO)


def build_train_tensors():
    img_mask_x = torch.from_numpy(
        np.random.randint(0, rows, sampled_pixel_count))
    img_mask_y = torch.from_numpy(
        np.random.randint(0, cols, sampled_pixel_count))

    img_train = img_ground_truth[img_mask_x, img_mask_y]

    img_mask_x = img_mask_x.float() / rows
    img_mask_y = img_mask_y.float() / cols

    img_mask = torch.stack([img_mask_x, img_mask_y], dim=-1)

    return img_mask, img_train


img_mask, img_train = build_train_tensors()

train_dataset = TensorDataset(img_mask, img_train)
train_dataloader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

# Build model
layers = [256, 256, 256, 256, 256]
in_features = 2
# Ben changed out_features from 3 to 4 don't know why but that fixed it lmao
out_features = 3
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

# TODO scheduler
# learning_rate = tf.keras.optimizers.schedules.PolynomialDecay(
#   0.0005, decay_steps=num_steps, end_learning_rate=5e-5, power=2.0)
# TODO: tensorboard

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = torch.nn.MSELoss()

checkpoint_dir = 'checkpoints/siren/inpainting/'
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

timestamp = datetime.now().strftime('%Y-%m-%d %H-%M-%S')
logdir = os.path.join('logs/siren/inpainting/', timestamp)

if not os.path.exists(logdir):
    os.makedirs(logdir)

set_logger(os.path.join(logdir, 'train.log'))

epoch_losses = []
epoch_improvements = []
best_loss = np.inf

before = datetime.now()

for epoch in range(EPOCHS):
    iterator = tqdm(train_dataloader, dynamic_ncols=True)

    losses = []

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

    if avg_loss < best_loss:
        logging.info('Loss improved from {:.4f} to {:.4f}'.format(
            best_loss, avg_loss))
        best_loss = avg_loss
        epoch_improvements.append([epoch,best_loss])
        torch.save(
            {'network': model.state_dict()},
            os.path.join(checkpoint_dir + 'model'))

complete = datetime.now()
dif = complete - before
before_str = before.strftime("%H:%M:%S")
logging.info('Started at: ' + before_str)
complete_str = complete.strftime("%H:%M:%S")
logging.info('Completed at: ' + complete_str)
logging.info('Total time: ' + str(dif.seconds//3600) + ':' + str((dif.seconds//60) % 60) + ":" + str(dif.seconds%60))

torch.save(model.state_dict(), "saved models/"+filename+" model.pt")

with open('metrics/'+filename+' - avg losses.csv','w+') as losses_file:
    writer = csv.writer(losses_file)
    writer.writerows(epoch_losses)

with open('metrics/'+filename+' - improvements.csv','w+') as improvements_file:
    writer = csv.writer(improvements_file)
    writer.writerows(epoch_improvements)

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




