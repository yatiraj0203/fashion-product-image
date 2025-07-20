import torch.nn as nn

class MultiOutputModel(nn.Module):
    def __init__(self, backbone, n_colours, n_types, n_seasons, n_genders):
        super().__init__()
        self.backbone = backbone
        self.backbone.fc = nn.Identity()
        self.fc = nn.Linear(512, 512)

        self.colour_head = nn.Linear(512, n_colours)
        self.type_head = nn.Linear(512, n_types)
        self.season_head = nn.Linear(512, n_seasons)
        self.gender_head = nn.Linear(512, n_genders)

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        return {
            'colour': self.colour_head(x),
            'product_type': self.type_head(x),
            'season': self.season_head(x),
            'gender': self.gender_head(x)
        }
