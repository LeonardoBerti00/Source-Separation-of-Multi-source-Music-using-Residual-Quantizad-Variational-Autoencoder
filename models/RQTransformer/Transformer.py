import torch
import torch.nn as nn
from torch.nn import functional as F
from einops import rearrange


class TransformerEncoder(nn.Module):
    def __init__(self, 
                num_heads: int, 
                d_model: int,  
                num_layers: int, 
                dropout: float, 
                ):
        super(TransformerEncoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.layers = nn.ModuleList([TransformerBlock(d_model, num_heads, dropout) for _ in range(num_layers)])
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.layer_norm(x)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, dropout):
        super(TransformerBlock, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.PReLU(init=0.01),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )
        self.num_heads = num_heads
        self.d_model = d_model
        self.to_qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.to_out = nn.Linear(d_model, d_model, bias=False)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x, mask):
        # x.shape = B, L, D
        # mask.shape = L, L

        # firstly we compute q, k, v and divide them into heads
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b l (h j) -> b h l j', h=self.num_heads), (q, k, v))
        # Scale the query (q) by the square root of the dimensionality of the model (d_model)
        q = torch.mul(q, 1/(self.d_model ** 0.5))
        # Compute the dot product of the query (q) and key (k) to get the raw attention scores (e)
        e = torch.einsum('b h l j, b h k i -> b h l k', q, k)
        # Apply the mask to the raw attention scores (e)
        e_masked = torch.add(e, mask)
        # Apply the softmax function to the masked attention scores to get the attention weights (att)
        att = torch.nan_to_num(F.softmax(e_masked, dim=-1))
        # Multiply the attention weights (att) with the value (v) to get the output of the attention mechanism
        out_att = torch.einsum('b h l l, b h l j -> b h l j', att, v)
        out_att = rearrange(out_att, 'b h l j -> b l (h j)')
        # Pass the output through the output linear layer
        out_att = self.to_out(out_att)
        # Apply the residual connection and layer normalization
        out_att = self.layer_norm(out_att + x)
        # Pass the output through the MLP
        out = self.mlp(out_att)

        return self.layer_norm(out + out_att)


