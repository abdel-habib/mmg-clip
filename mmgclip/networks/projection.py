import torch
from torch import nn

class LinearProjectionLayer(nn.Module):
    '''
    Implementation of Linear Projection layer for contrastive learning.
    
    Args:
        embedding_dim (int): The input dimension.
        projection_dim (int): The output dimension.
    
    Returns:
        output (tensor): The output tensor from the model.
    '''
    def __init__(self, embedding_dim, projection_dim=512, dropout=0):
        super().__init__()
        self.layer = nn.Linear(embedding_dim, projection_dim, bias=False)

        # By default it should be true
        for param in self.layer.parameters():
            param.requires_grad = True
        
    def forward(self, x):
        '''
        Forward pass of the model.

        Args:
            x (tensor): Input tensor shape shape (B, C).

        Returns:
            output (tensor): The output tensor from the model.
        '''
        return self.layer(x)


class MultiLinearHead(nn.Module):
    def __init__(self, embedding_dim, projection_dim=[], dropout=0.5):
        super(MultiLinearHead, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.projection_dim = projection_dim
        self.layers = nn.ModuleList()

        # Add the first linear layer from embedding_dim to the first projection dimension
        self.layers.append(nn.Linear(embedding_dim, projection_dim[0]))

        # Add the rest of the linear layers
        for i in range(len(projection_dim) - 1):
            self.layers.append(nn.Linear(projection_dim[i], projection_dim[i + 1]))

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = self.relu(x)
                x = self.dropout(x)

        return x


class ProjectionHead(nn.Module):
    def __init__(self, embedding_dim, hidden_dims = [512, 256, 128], output_dim = 64, dropout_p=0.1, use_batchnorm=True):
        super(ProjectionHead, self).__init__()
        layers = []
        for i in range(len(hidden_dims)):
            if i == 0:
                layers.append(nn.Linear(embedding_dim, hidden_dims[i]))
            else:
                layers.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_dims[i]))
            layers.append(nn.ReLU())
            if dropout_p > 0:
                layers.append(nn.Dropout(p=dropout_p))
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        self.projection = nn.Sequential(*layers)

    def forward(self, x):
        x = self.projection(x)
        return x

class MLPProjectionHead(nn.Module):
    def __init__(self, embedding_dim, projection_dim, dropout= 0.5):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x
