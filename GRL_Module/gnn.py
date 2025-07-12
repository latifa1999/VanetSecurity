import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class AttentionBlock(nn.Module):
    """
    Self-attention mechanism for focusing on relevant features in the graph.
    """
    def __init__(self, dim, heads=4):
        super(AttentionBlock, self).__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        assert self.head_dim * heads == dim, "dim must be divisible by heads"
        
        # Projection matrices
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        
        self.layer_norm = nn.LayerNorm(dim)
        
    def forward(self, x):
        # Save residual for skip connection
        residual = x
        
        # Layer normalization
        x = self.layer_norm(x)
        
        batch_size = x.size(0)
        
        # Project queries, keys, values
        q = self.query(x).view(batch_size, -1, self.heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(batch_size, -1, self.heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(batch_size, -1, self.heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        
        # Apply attention weights to values
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.dim)
        
        # Final projection
        out = self.out(out)
        
        # Add skip connection
        return out + residual

class EnhancedResidualBlock(nn.Module):
    """
    Enhanced residual block with batch normalization, dropout, and gated activations
    for better gradient flow and feature extraction.
    """
    def __init__(self, in_dim, hidden_dim=None, dropout_rate=0.2):
        super(EnhancedResidualBlock, self).__init__()
        if hidden_dim is None:
            hidden_dim = in_dim
        
        # First convolutional block
        self.block1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU()  # GELU activation often works better than ReLU
        )
        
        # Second convolutional block
        self.block2 = nn.Sequential(
            nn.Linear(hidden_dim, in_dim),
            nn.BatchNorm1d(in_dim),
        )
        
        # Gating mechanism for adaptive feature weighting
        self.gate = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = nn.GELU()
    
    def forward(self, x):
        identity = x
        
        # First block
        out = self.block1(x)
        out = self.dropout(out)
        
        # Second block
        out = self.block2(out)
        
        # Gating mechanism to control information flow
        gate_values = self.gate(identity)
        out = gate_values * out + (1 - gate_values) * identity  # Gated residual connection
        
        out = self.activation(out)
        return out

class CustomGNNPolicy(BaseFeaturesExtractor):
    """
    Enhanced feature extractor for the GNN policy with attention mechanisms
    and improved residual connections.
    """
    
    def __init__(self, observation_space, features_dim=512):
        super(CustomGNNPolicy, self).__init__(observation_space, features_dim)
        
        # Calculate input dimension
        input_dim = observation_space.shape[0]
        
        # Define network architecture with advanced components
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, 512),  # Increased initial dimension
            nn.LayerNorm(512),
            nn.GELU()
        )
        
        # Enhanced residual blocks with different hidden dimensions
        self.res_block1 = EnhancedResidualBlock(512, 768, dropout_rate=0.25)
        self.res_block2 = EnhancedResidualBlock(512, 768, dropout_rate=0.25)
        
        # Add attention mechanism after residual blocks
        self.attention = AttentionBlock(512, heads=8)
        
        # Downsampling path with multiple branches for multi-scale feature extraction
        self.branch1 = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU()
        )
        
        self.branch2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.Dropout(0.3),
            nn.GELU()
        )
        
        # Merge branches
        self.merge = nn.Sequential(
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.Dropout(0.2),
            nn.GELU()
        )
        
        # Final feature extraction layer
        self.feature_layer = nn.Sequential(
            nn.Linear(512, features_dim),
            nn.LayerNorm(features_dim),
            nn.Dropout(0.15),
            nn.GELU()
        )
        
        # Enhanced output normalization
        self.layer_norm = nn.LayerNorm(features_dim)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(p=0.25)
        
        # Initialize weights for better convergence
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize weights with improved techniques for better training."""
        if isinstance(module, nn.Linear):
            # Orthogonal initialization with higher gain for better gradients
            nn.init.orthogonal_(module.weight, gain=1.414)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(self, observations):
        """
        Forward pass with multi-branch processing and attention mechanism.
        """
        batch_size = observations.size(0)
        
        # Initial feature extraction
        x = self.input_layer(observations)
        x = self.dropout(x)
        
        # Process through residual blocks
        x = self.res_block1(x)
        x = self.res_block2(x)
        
        # Reshape for attention if needed (assumes batch processing)
        if len(x.shape) == 2:  # If input is [batch_size, features]
            attn_input = x.unsqueeze(1)  # Add sequence dimension [batch_size, 1, features]
        else:
            attn_input = x
            
        # Apply attention (if batch dimension is present)
        try:
            attn_output = self.attention(attn_input)
            if len(attn_output.shape) == 3:  # If output is [batch_size, seq_len, features]
                x = attn_output.squeeze(1)  # Remove sequence dimension
            else:
                x = attn_output
        except:
            # Fallback in case of dimension issues with attention
            pass  # Keep x as is
        
        # Multi-branch processing for diverse feature capture
        branch1_out = self.branch1(x)
        branch2_out = self.branch2(x)
        
        # Concatenate branches and merge
        combined = torch.cat([branch1_out, branch2_out], dim=1)
        x = self.merge(combined)
        
        # Final feature extraction
        x = self.feature_layer(x)
        
        # Apply final normalization
        x = self.layer_norm(x)
        
        return x