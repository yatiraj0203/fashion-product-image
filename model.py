# model.py
import torch
import torch.nn as nn

class MultiOutputModel(nn.Module):
    def __init__(self, num_colors, num_product_types, num_seasons, num_genders):
        super(MultiOutputModel, self).__init__()
        self.base_model = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 112 * 112, 512),
            nn.ReLU(),
        )
        self.color_head = nn.Linear(512, num_colors)
        self.product_type_head = nn.Linear(512, num_product_types)
        self.season_head = nn.Linear(512, num_seasons)
        self.gender_head = nn.Linear(512, num_genders)

    def forward(self, x):
        features = self.base_model(x)
        return {
            'color': self.color_head(features),
            'product_type': self.product_type_head(features),
            'season': self.season_head(features),
            'gender': self.gender_head(features),
        }
