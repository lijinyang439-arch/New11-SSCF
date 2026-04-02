import torch
import torch.nn as nn
from .transformer import TransformerEncoder
from .encoder import Encoder
from .decoder import Decoder


class AE(nn.Module):
    def __init__(self, params):
        super(AE, self).__init__()
        self.encoder = Encoder(params)
        self.decoder = Decoder(params)

    def sample_z(self, mu, log_var):
        """sample z by reparameterization trick"""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, domain_ids=None):
        mu = self.encoder(x, domain_ids=domain_ids) # 传给 Encoder
        # z = self.sample_z(mu, log_var)
        recon = self.decoder(mu)
        return recon, mu