# Modified work based on the original code:
# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Original work licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
# See http://creativecommons.org/licenses/by-nc-sa/4.0/
#
# This modified version is also licensed under the same Creative Commons 
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# Modifications copyright (c) 2025, Petrus Mikkola, University of Helsinki

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


#EDM toy model
#https://github.com/NVlabs/edm2/blob/main/tojoint_example.py
    
#----------------------------------------------------------------------------
# Low-level primitives used by ToyModel.

def normalize(x, dim=None, eps=1e-4):
    if dim is None:
        dim = list(range(1, x.ndim))
    norm = torch.linalg.vector_norm(x, dim=dim, keepdim=True, dtype=torch.float32)
    norm = torch.add(eps, norm, alpha=np.sqrt(norm.numel() / x.numel()))
    return x / norm.to(x.dtype)

class MPSiLU(torch.nn.Module):
    def forward(self, x):
        return torch.nn.functional.silu(x) / 0.596

class MPLinear(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(out_dim, in_dim))

    def forward(self, x):
        if self.training:
            with torch.no_grad():
                self.weight.copy_(normalize(self.weight))
        w = normalize(self.weight) / np.sqrt(self.weight[0].numel())
        return x @ w.t()   
    
    
#----------------------------------------------------------------------------
# Denoiser model for learning 2D toy distributions.
    
class EDMToyModel(torch.nn.Module):
    def __init__(self,
        in_dim      = 4,    # Input dimensionality.
        num_layers  = 4,    # Number of hidden layers.
        hidden_dim  = 64,   # Number of hidden features.
        sigma_data  = 1.0,  # Expected standard deviation of the training data.
        use_temp = False    # Whether use temp variable at all. If not, temp should not harm training.
    ):
        super().__init__()
        
        assert hidden_dim % 4 == 0, "hidden_dim must be divisible by 4"
        
        self.sigma_data = sigma_data

        self.use_temp = use_temp

        self.joint_embed = MPLinear(1, hidden_dim // 4)
        self.joint_residual = MPLinear(hidden_dim // 4, hidden_dim)
        self.temp_embed = MPLinear(1, hidden_dim // 4)
        self.temp_residual = MPLinear(hidden_dim // 4, hidden_dim)

        self.silu = MPSiLU()

        self.input_layer = MPLinear(in_dim + 1 + (hidden_dim // 2), hidden_dim)
        self.hidden_layers = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.hidden_layers.append(nn.ModuleList([
                MPSiLU(),
                MPLinear(hidden_dim, hidden_dim)
            ]))
        self.output_layer = MPLinear(hidden_dim, in_dim)
        self.gain = torch.nn.Parameter(torch.zeros([]))


    def forward(self, x, sigma, joint, temp):

        #x: float tensor of shape (n,d)
        #sigma: float tensor of shape (n,1)
        #joint: binary tensor of shape (n,1)
        #temp: binary tensor of shape (n,1)

        batch_size = x.shape[0]
        def to_column_vector(t):
            t = torch.as_tensor(t, dtype=torch.float32, device=x.device)
            if t.ndim == 0:
                t = t.expand(batch_size, 1)
            elif t.ndim == 1:
                t = t.unsqueeze(-1)
            return t
        sigma = to_column_vector(sigma)
        temp = to_column_vector(temp)
        joint = to_column_vector(joint)
        
        c_input = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt() 
        c_output = (sigma * self.sigma_data) / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_skip = (self.sigma_data ** 2) / (self.sigma_data ** 2 + sigma ** 2)
        c_noise = sigma.log() / 4
        
        #center boolean variables 0/1 to avoid large unconditional bias
        joint_c = joint - 0.5
        temp_c  = temp - 0.5 if self.use_temp else torch.zeros_like(temp)

        joint_emb = self.silu(self.joint_embed(joint_c))
        temp_emb  = self.silu(self.temp_embed(temp_c))

        h = self.input_layer(torch.cat([c_input*x, c_noise, joint_emb, temp_emb], dim=-1))

        joint_residual = self.joint_residual(joint_emb)
        temp_residual  = self.temp_residual(temp_emb)
        res = joint_residual + temp_residual

        #prevents the 1/sigma amplification in score
        alpha = sigma / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        res = res * alpha.expand_as(res) / len(self.hidden_layers)

        #simple residual connection for joint and temp
        for activation, linear in self.hidden_layers:
            h = activation(h)
            h = linear(h)
            h = h + res

        F_net = self.output_layer(h) * self.gain.exp()

        denoised_x = c_skip*x + c_output*F_net
        score = (denoised_x - x) / (sigma**2)
        
        return score