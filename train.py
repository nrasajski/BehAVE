import os
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from text_encoders import PretrainedEmbeddingClip
from tqdm import tqdm
from pathlib import Path

from dataset import BehAVEDataset
from model import BehAVE

# Check if CUDA is available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Hyperparameters
INPUT_EMB_SIZE = 768
OUTPUT_EMB_SIZE = 512
LEARNING_RATE = 1e-3
BATCH_SIZE = 128
NUM_EPOCHS = 1
ROOT_DIR = r'C:\Datasets\Actions_FPS\bench'
CHECKPOINT_SAVE_PATH = "check.pt"
INIT_FROM_CHECKPOINT = False

text_encoder = PretrainedEmbeddingClip(device=device)

game_folders = [os.path.join(ROOT_DIR, folder) for folder in os.listdir(ROOT_DIR) if
                os.path.isdir(os.path.join(ROOT_DIR, folder))]
game_folders_train = random.sample(game_folders, 15)
train_dataset = BehAVEDataset(root_dirs=game_folders_train, text_encoder=text_encoder, train=True, split_pc=1)
train_data_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)


# Initialize the joint embedding model, loss function, and optimizer
model = BehAVE(OUTPUT_EMB_SIZE, INPUT_EMB_SIZE).to(device)
if INIT_FROM_CHECKPOINT and os.path.isfile(CHECKPOINT_SAVE_PATH):
    model.load_state_dict(torch.load(CHECKPOINT_SAVE_PATH))
    model = model.to(device)
    model.train()

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
cosine_criterion = nn.CosineEmbeddingLoss().to(device)

for epoch in range(NUM_EPOCHS):
    loader_pbar = tqdm(enumerate(train_data_loader), unit="batch", total=len(train_data_loader))
    for batch_idx, (video_latents, action_embeddings, action_string) in loader_pbar:
        video_latents, action_embeddings = video_latents.to(device).squeeze(1), action_embeddings.to(device).squeeze(
            1).type(torch.float32)

        # Forward pass
        video_embeddings, text_embeddings = model(video_latents, action_embeddings)

        # Cosine loss
        target_cosine = torch.ones(BATCH_SIZE).to(device)  # CosineEmbeddingLoss requires 1 for similarity
        loss = cosine_criterion(video_embeddings, text_embeddings, target_cosine)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loader_pbar.set_description(f"Epoch {epoch+1} Training Loss: {loss.item():.4f}")

torch.save(model.state_dict(), CHECKPOINT_SAVE_PATH)
print(f"Training finished. Model saved to: {Path(CHECKPOINT_SAVE_PATH)}")

