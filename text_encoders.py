from transformers import AutoTokenizer, AutoModel
import torch.nn as nn
import torch
import clip
from sentence_transformers import SentenceTransformer


class PretrainedEmbeddingHF(nn.Module):
    def __init__(self, pretrained_model_name, device: str = "cpu"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
        self.model = AutoModel.from_pretrained(pretrained_model_name)
        self.device = device

    def forward(self, text):
        action_token = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = self.model(**action_token)
            action_embedding = output.last_hidden_state.mean(dim=1).squeeze(0).cpu()
        return action_embedding


class PretrainedEmbeddingClip(nn.Module):
    def __init__(self, pretrained_model_name: str = "ViT-B/32", device: str = "cpu"):
        super().__init__()
        self.clip_model, _ = clip.load(pretrained_model_name, device=device)
        self.device = device

    def forward(self, text):
        action_token = clip.tokenize(text, truncate=True).to(self.device)
        with torch.no_grad():
            action_embedding = self.clip_model.encode_text(action_token)
        return action_embedding


class PretrainedEmbeddingSTransformer(nn.Module):
    def __init__(self, pretrained_model_name, device: str = "cpu"):
        super().__init__()
        self.model = SentenceTransformer(pretrained_model_name).to(device)
        self.device = device

    def forward(self, text):
        action_embedding = torch.from_numpy(self.model.encode(text))
        return action_embedding
