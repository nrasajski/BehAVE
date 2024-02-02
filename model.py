from torch import nn


class BehAVE(nn.Module):
    def __init__(self,
                 output_emb_size=512,
                 input_emb_size=768,
                 hidden_size=1024,
                 dropout=0.5,
                 normalize=True):
        super(BehAVE, self).__init__()
        self.normalize = normalize
        self.projection = nn.Sequential(
            nn.Linear(input_emb_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 2 * hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_emb_size)
        )

    def forward(self, video_latent, action_embedding):
        if self.normalize:
            # increasing size of hypersphere as proposed in https://arxiv.org/abs/2206.09616
            projected_video_features = nn.functional.normalize(self.projection(video_latent), p=2, dim=1) * 8
            if action_embedding is not None:
                projected_text_features = nn.functional.normalize(action_embedding, p=2, dim=1) * 8
            else:
                projected_text_features = action_embedding
        else:
            projected_video_features = self.projection(video_latent)
            projected_text_features = action_embedding

        return projected_video_features, projected_text_features
