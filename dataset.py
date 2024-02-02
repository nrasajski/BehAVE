import os
import random

import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm


class BehAVEDataset(Dataset):
    def __init__(self,
                 root_dirs,
                 text_encoder,
                 train=True,
                 split_seed=117,
                 split_pc=0.8):
        self.root_dirs = root_dirs
        self.latent_files = []

        self.text_sequence = ['Pan Left', 'Pan Right', 'Pan Up', 'Pan Down', 'Fire Gun', 'Aim Gun', 'Move Forward',
                              'Strafe Left', 'Move Backward', 'Strafe Right', 'Jump', 'Crouch', 'Sprint',
                              'Change Gun', 'Change Gun', 'Change Gun', 'Reload Gun', 'Interact', 'Crouch']

        # Lists to store action embeddings and action strings
        self.action_embeddings = []
        self.action_strings = []
        self.text_encoder = text_encoder

        for root_dir in tqdm(root_dirs, desc="Loading data from all games"):
            game_latent_files = [file for file in os.listdir(os.path.join(root_dir, 'gifs', 'latents_stride_8')) if
                                 file.endswith('.npy')]
            game_actions_df = pd.read_csv(os.path.join(root_dir, 'gifs', 'gif_actions_stride_8.csv'))
            # Add game name column to the actions dataframe
            game_name = os.path.basename(root_dir)
            game_actions_df['Game'] = game_name
            # Randomly shuffle the indices
            random.seed(split_seed)
            random.shuffle(game_latent_files)
            if train:
                game_latent_files = game_latent_files[:int(split_pc * len(game_latent_files))]
            else:
                game_latent_files = game_latent_files[int(split_pc * len(game_latent_files)):]

            self.latent_files.extend(
                [os.path.join(root_dir, 'gifs', 'latents_stride_8', file) for file in game_latent_files])

            for file in [os.path.join(root_dir, 'gifs', 'latents_stride_8', file) for file in game_latent_files]:
                action_row = game_actions_df.loc[
                                 (game_actions_df['Image Filename'] == os.path.basename(file).replace('.npy', '.gif'))
                             ].iloc[:, :-1]
                actions = action_row.values[0, 1:].astype(np.float32)  # Exclude the image filename from actions
                actions_list = [self.text_sequence[i] for i, label in enumerate(actions) if label == 1]
                action_string = ", ".join(actions_list) if actions_list else "Inaction"
                # self.action_embeddings.append(actions)
                self.action_strings.append(action_string)

        # Dictionary to store pre-calculated embeddings for unique action_strings
        self.embedding_dict = {}
        unique_action_strings = set(self.action_strings)
        for unique_action_string in tqdm(unique_action_strings, desc="Extracting Text action embeddings"):
            action_embedding = self.text_encoder(unique_action_string)
            self.embedding_dict[unique_action_string] = action_embedding

    def __len__(self):
        return len(self.latent_files)

    def __getitem__(self, idx):
        latent_file = self.latent_files[idx]
        action_string = self.action_strings[idx]
        action_embedding = self.embedding_dict[action_string]
        latent_data = np.load(latent_file)  # Load latent data
        return latent_data, action_embedding, action_string
