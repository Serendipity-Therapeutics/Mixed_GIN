# models/gin_infomax.py

import numpy as np
import torch
import torch.nn as nn
from molfeat.trans.pretrained import PretrainedDGLTransformer

class GINBinaryClassifier(nn.Module):
    def __init__(self, transformer_kind='gin_supervised_infomax', hidden_size=256):
        super(GINBinaryClassifier, self).__init__()

        self.transformer = PretrainedDGLTransformer(kind=transformer_kind, dtype=float)

        feature_dim = 300
        
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid() 
        )

    def forward(self, smiles_list):
        features = self.transformer(smiles_list)

        if isinstance(features, np.ndarray):
            device = next(self.parameters()).device
            features = torch.tensor(features, dtype=torch.float32, device=device)

        output = self.classifier(features)
        return output


class GINRegressor(nn.Module):
    def __init__(self, transformer_kind='gin_supervised_infomax', hidden_size=256):
        super(GINRegressor, self).__init__()
        
        self.transformer = PretrainedDGLTransformer(kind=transformer_kind, dtype=float)
        
        feature_dim = 300
        
        self.regressor = nn.Sequential(
            nn.Linear(feature_dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 1) 
        )

    def forward(self, smiles_list):
        features = self.transformer(smiles_list)

        if isinstance(features, np.ndarray):
            device = next(self.parameters()).device
            features = torch.tensor(features, dtype=torch.float32, device=device)

        output = self.regressor(features)
        return output