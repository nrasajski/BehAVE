import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import clip
import numpy as np
import pandas as pd
import random
import datetime
from time import time
from model import ActionVideoAlignment
from src.datasets.alignment_dataset import AlignmentDataset, AlignmentDatasetExpandedActions
from src.utils.csv_action_utls import load_actions_csv
from src.models.text_embeddings import PretrainedEmbeddingHF, PretrainedClipEmbedding
from sklearn.manifold import TSNE
from bokeh.models import ColumnDataSource, HoverTool, ColorBar
import matplotlib.pyplot as plt
from bokeh.plotting import figure, show, output_file
from transformers import AutoTokenizer, AutoModel, AlignTextModel, Data2VecTextModel, FlavaTextModel, OpenAIGPTModel, GPT2Model, GPTNeoModel, MobileBertModel, BertModel, MPNetModel

from tqdm import tqdm
from sklearn.metrics import silhouette_score
from src.utils.custom_loss_functions import spearman
from sentence_transformers import SentenceTransformer

model.eval()
all_video_features = []
all_videomae_latents = []
all_video_descs = []
all_video_filepaths = []
with torch.no_grad():
    total_cosine_similarity = 0
    instance_counter = 0
    sim = nn.CosineSimilarity(dim=1)
    for batch_idx, (video_latents, action_embeddings, action_strings, gif_filepaths) in enumerate(
            test_data_loader):
        video_latents, action_embeddings = video_latents.to(device).squeeze(1), action_embeddings.to(
            device).squeeze(1).type(torch.float32)
        video_embeddings, text_embeddings = model(video_latents, action_embeddings)

        all_videomae_latents.append(nn.functional.normalize(video_latents, p=2, dim=1).cpu().numpy())
        all_video_features.append(video_embeddings.cpu().numpy())
        all_video_descs.extend(action_strings)
        all_video_filepaths.extend(gif_filepaths)

        # all_video_features.append(nn.functional.normalize(action_embeddings, p=2, dim=1).cpu().numpy())


all_video_features = np.concatenate(all_video_features, axis=0)
all_videomae_latents = np.concatenate(all_videomae_latents, axis=0)

game_labels = list(
    map(lambda path: os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(path)))), all_video_filepaths))
weapon_labels = ['Weapon' if 'Gun' in l else 'No Weapon' for l in all_video_descs]
navigation_labels = ['Movement' if 'Move' in l or 'Strafe' in l else 'No Movement' for l in all_video_descs]
label_text = ['gamename', 'weapon', 'navigation', 'behavior', 'panning']
classification_labels = [
    'Weapon, Navigation' if 'Gun' in l and ('Move' in l or 'Strafe' in l) else
    'Weapon, No Navigation' if 'Gun' in l and not ('Move' in l or 'Strafe' in l) else
    'No Weapon, Navigation' if 'Gun' not in l and ('Move' in l or 'Strafe' in l) else
    'No Weapon, No Navigation'
    for l in all_video_descs
]
panning_labels = ['Panning' if 'Pan' in l else 'No Panning' for l in all_video_descs]


# Perform silhouette score calculations
weapon_videomae_score = silhouette_score(all_videomae_latents, weapon_labels, metric='cosine')
weapon_aligned_score = silhouette_score(all_video_features, weapon_labels, metric=silhouette_metric)
navigation_videomae_score = silhouette_score(all_videomae_latents, navigation_labels, metric=silhouette_metric)
navigation_aligned_score = silhouette_score(all_video_features, navigation_labels, metric=silhouette_metric)
gamecluster_videomae_score = silhouette_score(all_videomae_latents, game_labels, metric=silhouette_metric)
gamecluster_aligned_score = silhouette_score(all_video_features, game_labels, metric=silhouette_metric)
panning_videomae_score = silhouette_score(all_videomae_latents, panning_labels, metric=silhouette_metric)
panning_aligned_score = silhouette_score(all_video_features, panning_labels, metric=silhouette_metric)
weaponnavigation_videomae_score = silhouette_score(all_videomae_latents, classification_labels, metric=silhouette_metric)
weaponnavigation_aligned_score = silhouette_score(all_video_features, classification_labels, metric=silhouette_metric)

# Save silhouette scores to a text file
silhouette_scores_filename = "artifacts/silhouette_tests/silhouette_scores.txt"
with open(silhouette_scores_filename, 'w') as file:
    file.write(f'Weapon (videomae): {weapon_videomae_score}\n')
    file.write(f'Weapon (aligned): {weapon_aligned_score}\n')
    file.write(f'Navigation (videomae): {navigation_videomae_score}\n')
    file.write(f'Navigation (aligned): {navigation_aligned_score}\n')
    file.write(f'Weapon+Navigation (videomae): {weaponnavigation_videomae_score}\n')
    file.write(f'Weapon+Navigation (aligned): {weaponnavigation_aligned_score}\n')
    file.write(f'Game Cluster (videomae): {gamecluster_videomae_score}\n')
    file.write(f'Game Cluster (aligned): {gamecluster_aligned_score}\n')
    file.write(f'Panning (videomae): {panning_videomae_score}\n')
    file.write(f'Panning (aligned): {panning_aligned_score}\n')

# Print the file path for confirmation
print(f'Silhouette scores saved to: {silhouette_scores_filename}')
print('weapon_videomae_score', weapon_videomae_score)
print('weapon_aligned_score', weapon_aligned_score)
print('navigation_videomae_score', navigation_videomae_score)
print('navigation_aligned_score', navigation_aligned_score)
print('weaponnavigation_videomae_score', weaponnavigation_videomae_score)
print('weaponnavigation_aligned_score', weaponnavigation_aligned_score)
print('gamecluster_videomae_score', gamecluster_videomae_score)
print('gamecluster_aligned_score', gamecluster_aligned_score)
print('panning_videomae_score', panning_videomae_score)
print('panning_aligned_score', panning_aligned_score)