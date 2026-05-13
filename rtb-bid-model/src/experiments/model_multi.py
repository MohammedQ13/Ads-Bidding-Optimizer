"""Multi-head model: shared backbone, MDN + DiscreteBins heads."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import CategoricalEmbeddings, MLPBackbone


class MultiHead(nn.Module):
    def __init__(self, vocab_sizes, emb_dims, tag_vocab_size, tag_emb_dim,
                 num_continuous, hidden, dropout, K, sigma_floor, num_bins):
        super().__init__()
        self.emb = CategoricalEmbeddings(vocab_sizes, emb_dims, tag_vocab_size, tag_emb_dim)
        self.cont_norm = nn.Identity()
        in_dim = self.emb.out_dim + num_continuous
        self.backbone = MLPBackbone(in_dim, hidden, dropout)
        self.K = K
        self.sigma_floor = sigma_floor
        self.num_bins = num_bins
        self.head_mdn = nn.Linear(self.backbone.out_dim, 3 * K)
        self.head_bins = nn.Linear(self.backbone.out_dim, num_bins)

    def forward(self, cat, cont, tags):
        e = self.emb(cat, tags)
        c = self.cont_norm(cont)
        x = torch.cat([e, c], dim=1)
        h = self.backbone(x)
        m = self.head_mdn(h)
        pi_logits = m[:, :self.K]
        mu = m[:, self.K:2 * self.K]
        log_sigma = m[:, 2 * self.K:]
        sigma = F.softplus(log_sigma) + self.sigma_floor
        bins_logits = self.head_bins(h)
        return pi_logits, mu, sigma, bins_logits
