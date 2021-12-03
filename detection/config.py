import torch
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

BATCH_SIZE = 2 # increase / decrease according to GPU memeory
RESIZE_TO = 1000 # resize the image for training and transforms
NUM_EPOCHS = 30 # number of epochs to train for
NUM_WORKERS = 1

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# training images and XML files directory
TRAIN_DIR = BASE_DIR / 'data/train'
# validation images and XML files directory
VALID_DIR = BASE_DIR / 'data/valid'

# classes: 0 index is reserved for background
# Example: CLASSES = ['__background__', "Car", "Person", "Flower" .....]
CLASSES = [
    '__background__',
]

NUM_CLASSES = len(CLASSES)

# name to save the trained model with
MODEL_NAME = 'model'

# whether to visualize images after crearing the data loaders
VISUALIZE_TRANSFORMED_IMAGES = False

VISUALIZE_PREDICTED_IMAGES = False

# location to save model and plots
OUT_DIR = BASE_DIR / 'outputs'