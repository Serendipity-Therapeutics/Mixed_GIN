# Gin_MoE_model.py

import math
import numpy as np
import torch
import torch.nn as nn
from molfeat.trans.pretrained import PretrainedDGLTransformer

class PretrainedTransformerWrapper(nn.Module):
    def __init__(self, kind, dtype=float):
        super(PretrainedTransformerWrapper, self).__init__()
        self.transformer = PretrainedDGLTransformer(kind=kind, dtype=dtype)

    def forward(self, smiles_list):
        return self.transformer(smiles_list)

class AdvancedMoE_GINClassifier(nn.Module):

    def __init__(self, transformer_kinds, feature_dim=300, hidden_size=256, expert_dropout=0.1, gating_hidden_dim=None):

        super(AdvancedMoE_GINClassifier, self).__init__()
        self.num_experts = len(transformer_kinds)
        self.feature_dim = feature_dim

        self.experts = nn.ModuleList([
            PretrainedTransformerWrapper(k, dtype=float) for k in transformer_kinds
        ])

        if gating_hidden_dim is None:
            gating_hidden_dim = (self.num_experts * feature_dim) // 2
        self.gating = nn.Sequential(
            nn.Linear(self.num_experts * feature_dim, gating_hidden_dim),
            nn.GELU(),
            nn.Dropout(expert_dropout),
            nn.Linear(gating_hidden_dim, self.num_experts)
        )

        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.GELU(),
            nn.Dropout(expert_dropout),
            nn.Linear(hidden_size, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(expert_dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(expert_dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if m.weight is not None:
                    fan_in = nn.init._calculate_correct_fan(m.weight, mode='fan_in')
                    std = 1.0 / math.sqrt(fan_in)
                    with torch.no_grad():
                        m.weight.normal_(0, std)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
    def forward(self, smiles_list):
        device = next(self.parameters()).device
        expert_features = []
        for expert in self.experts:
            features = expert(smiles_list)
            if isinstance(features, np.ndarray):
                features = torch.tensor(features, dtype=torch.float32, device=device)
            expert_features.append(features)
        expert_features_stack = torch.stack(expert_features, dim=1) 
        gating_input = expert_features_stack.view(expert_features_stack.size(0), -1) 
        gating_logits = self.gating(gating_input) 
        gating_weights = torch.softmax(gating_logits, dim=1).unsqueeze(2)  
        fused_feature = torch.sum(gating_weights * expert_features_stack, dim=1) 
        output = self.classifier(fused_feature)
        return output

class AdvancedMoE_GINRegressor(nn.Module):

    def __init__(self, transformer_kinds, feature_dim=300, hidden_size=256, expert_dropout=0.1, gating_hidden_dim=None):
        super(AdvancedMoE_GINRegressor, self).__init__()
        self.num_experts = len(transformer_kinds)
        self.feature_dim = feature_dim

        self.experts = nn.ModuleList([
            PretrainedTransformerWrapper(k, dtype=float) for k in transformer_kinds
        ])

        if gating_hidden_dim is None:
            gating_hidden_dim = (self.num_experts * feature_dim) // 2
        self.gating = nn.Sequential(
            nn.Linear(self.num_experts * feature_dim, gating_hidden_dim),
            nn.GELU(),
            nn.Dropout(expert_dropout),
            nn.Linear(gating_hidden_dim, self.num_experts)
        )

        self.regressor = nn.Sequential(
            nn.Linear(feature_dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.GELU(),
            nn.Dropout(expert_dropout),
            nn.Linear(hidden_size, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(expert_dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(expert_dropout),
            nn.Linear(64, 1)
        )
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if m.weight is not None:
                    fan_in = nn.init._calculate_correct_fan(m.weight, mode='fan_in')
                    std = 1.0 / math.sqrt(fan_in)
                    with torch.no_grad():
                        m.weight.normal_(0, std)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
    def forward(self, smiles_list):
        device = next(self.parameters()).device
        expert_features = []
        for expert in self.experts:
            features = expert(smiles_list)
            if isinstance(features, np.ndarray):
                features = torch.tensor(features, dtype=torch.float32, device=device)
            expert_features.append(features)
        expert_features_stack = torch.stack(expert_features, dim=1)
        gating_input = expert_features_stack.view(expert_features_stack.size(0), -1)
        gating_logits = self.gating(gating_input)
        gating_weights = torch.softmax(gating_logits, dim=1).unsqueeze(2)
        fused_feature = torch.sum(gating_weights * expert_features_stack, dim=1)
        output = self.regressor(fused_feature)
        return output