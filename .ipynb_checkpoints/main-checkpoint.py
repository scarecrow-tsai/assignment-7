import os
import random
from utils.test_loop import test_loop
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

from utils.load_model import ModelCL
from utils.train_loop import train_loop
from utils.load_data import load_dataset, create_samplers


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


set_seed(100)


################################
## CONFIG
################################
DATASET_NAME = "cifar10"
DATASET_ROOT_PATH = f"./../data/{DATASET_NAME}/"
NUM_CLASSES = 10

BATCH_SIZE = 128
EPOCHS = 15
LEARNING_RATE = 0.001


# SET GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("\nWe're using =>", device)


################################
## LOAD DATASET
################################


# transforms
image_transforms = {
    "train": A.Compose(
        [
            A.ShiftScaleRotate(
                shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=0.5
            ),
            A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    ),
    "test": A.Compose(
        [
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    ),
}

# datasets
train_dataset = load_dataset(
    dataset_name=DATASET_NAME,
    dataset_root_path=DATASET_ROOT_PATH,
    is_train=True,
    image_transforms=image_transforms["train"],
)

test_dataset = load_dataset(
    dataset_name=DATASET_NAME,
    dataset_root_path=DATASET_ROOT_PATH,
    is_train=False,
    image_transforms=image_transforms["test"],
)

# train-val sampler
train_sampler, val_sampler = create_samplers(train_dataset, 0.8)


# dataloader
train_loader = DataLoader(
    dataset=train_dataset, shuffle=False, batch_size=BATCH_SIZE, sampler=train_sampler,
)

val_loader = DataLoader(
    dataset=train_dataset, shuffle=False, batch_size=1, sampler=val_sampler
)

test_loader = DataLoader(dataset=test_dataset, shuffle=False, batch_size=1)

## Data Sanity Check
print(f"\nTrain loader = {next(iter(train_loader))[0].shape}")
print(f"Val loader = {next(iter(val_loader))[0].shape}")
print(f"Test loader = {next(iter(test_loader))[0].shape}")
print(f"\nTrain loader length = {len(train_loader)}")
print(f"Val loader length = {len(val_loader)}")
print(f"Test loader length = {len(test_loader)}")


################################
## LOAD MODEL
################################
model = ModelCL(num_classes=NUM_CLASSES, norm_type="bnorm")

x_train_example, y_train_example = next(iter(train_loader))
y_pred_example = model(x_train_example)

print("\nShape of output pred = ", y_pred_example.shape)

model.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

################################
## Train Loop
################################
trained_model, loss_stats, acc_stats = train_loop(
    model=model,
    epochs=EPOCHS,
    optimizer=optimizer,
    criterion=criterion,
    scheduler=scheduler,
    train_loader=train_loader,
    val_loader=val_loader,
    device=device,
)


################################
## Test Loop
################################
y_pred_list, y_true_list = test_loop(
    model=trained_model, test_loader=test_loader, device=device,
)
