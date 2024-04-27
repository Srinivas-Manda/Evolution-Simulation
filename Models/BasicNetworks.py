import torch
import torch.nn as nn
import torch.nn.functional as F

# the feature extractor that gives the state representation
class CNN(nn.Module):
    def __init__(self, config, out_features):
        super().__init__()
        
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=config['num_channels'], out_channels=8, kernel_size=1, padding=0),
            # nn.Dropout(p=0.3),
            nn.BatchNorm2d(num_features=8),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1),
            # nn.Dropout(p=0.3),
            nn.BatchNorm2d(num_features=16),
            nn.Conv2d(in_channels=16, out_channels=config["out_features"], kernel_size=3, padding=1),
            # nn.Dropout(p=0.3),
            nn.BatchNorm2d(num_features=config["out_features"]),
        )
        
        self.avg_pool = nn.AdaptiveMaxPool2d(1)
        
    def forward(self, state):
        feature_map = self.conv_block(state)
        x = self.avg_pool(feature_map).flatten(-3)
        
        return x
    
    
def create_linear_network(input_dim, output_dim, hidden_dims=[], dropout_prob=0.3):
    model = []
    units = input_dim
    for next_units in hidden_dims:
        model.append(nn.Linear(units, next_units))
        model.append(nn.ReLU())
        model.append(nn.Dropout(p=dropout_prob))
        units = next_units

    model.append(nn.Linear(units, output_dim))
    model.append(nn.ReLU())

    return nn.Sequential(*model)

def create_convolutional_network(input_channels, output_channels, hidden_channels=[], dropout_prob=0.3):
    model = []
    channels = input_channels
    for next_channels in hidden_channels:
        model.append(nn.Conv2d(in_channels=channels, out_channels=next_channels, kernel_size=3, stride=1))
        model.append(nn.BatchNorm2d(next_channels))
        model.append(nn.ReLU())
        # model.append(nn.Dropout(p=dropout_prob))
        
        channels = next_channels
        
    model.append(nn.Conv2d(in_channels=channels, out_channels=output_channels, kernel_size=3, stride=1))
    model.append(nn.BatchNorm2d(output_channels))
    model.append(nn.ReLU())
    model.append(nn.AdaptiveAvgPool2d(1))

    return nn.Sequential(*model)

# MLP block
class MLP(nn.Module):
    def __init__(self, config, out_dim):
        self.mlp_block = nn.Sequential(
            nn.Linear(in_features=16, out_features=32),
            # nn.Dropout(p=0.3),
            nn.Linear(in_features=32, out_features=config['num_actions'])
        )