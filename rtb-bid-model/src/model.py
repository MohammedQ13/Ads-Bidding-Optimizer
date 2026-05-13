import torch
import torch.nn as nn
import torch.nn.functional as F


class CategoricalEmbeddings(nn.Module):
    """Embedding layer for categorical features and user tags."""

    def __init__(self, vocab_sizes, emb_dims, tag_vocab_size, tag_emb_dim):
        super().__init__()
        # one embedding table per categorical feature (region, city, domain, etc)
        self.embs = nn.ModuleList()
        self.feat_names = list(vocab_sizes.keys())
        out_dim = 0
        for name in self.feat_names:
            self.embs.append(nn.Embedding(vocab_sizes[name], emb_dims[name]))
            out_dim += emb_dims[name]
        # separate embedding for user tags since each impression has multiple tags
        self.tag_emb = nn.Embedding(tag_vocab_size, tag_emb_dim, padding_idx=0)
        self.tag_dim = tag_emb_dim
        self.out_dim = out_dim + tag_emb_dim

    def forward(self, cat, tags):
        # look up embedding for each categorical feature
        parts = []
        for i, e in enumerate(self.embs):
            parts.append(e(cat[:, i]))
        # tags: (B, 10) where each row has up to 10 tag IDs, padded with 0
        # embed each tag then mean-pool to get a single vector per sample
        tag_e = self.tag_emb(tags)  # (B, 10, D)
        mask = (tags != 0).float().unsqueeze(-1)  # (B, 10, 1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        bag = (tag_e * mask).sum(dim=1) / denom
        parts.append(bag)
        return torch.cat(parts, dim=1)


class MLPBackbone(nn.Module):
    """Shared MLP backbone. Linear -> LayerNorm -> GELU -> Dropout per layer."""

    def __init__(self, in_dim, hidden, dropout):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.LayerNorm(h))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            prev = h
        self.net = nn.Sequential(*layers)
        self.out_dim = prev

    def forward(self, x):
        return self.net(x)


class MDN(nn.Module):
    """Mixture density network. Outputs K Gaussian components in log-price space.
    Head outputs 3*K values: K weights (pi), K means (mu), K sigmas.
    """

    def __init__(self, vocab_sizes, emb_dims, tag_vocab_size, tag_emb_dim,
                 num_continuous, hidden, dropout, K, sigma_floor):
        super().__init__()
        self.emb = CategoricalEmbeddings(vocab_sizes, emb_dims, tag_vocab_size, tag_emb_dim)
        self.cont_norm = nn.Identity()
        in_dim = self.emb.out_dim + num_continuous
        self.backbone = MLPBackbone(in_dim, hidden, dropout)
        self.K = K
        # sigma_floor prevents sigma from collapsing to zero
        # which would make the NLL loss explode
        self.sigma_floor = sigma_floor
        self.head = nn.Linear(self.backbone.out_dim, 3 * K)

    def forward(self, cat, cont, tags):
        e = self.emb(cat, tags)
        c = self.cont_norm(cont)
        x = torch.cat([e, c], dim=1)
        h = self.backbone(x)
        out = self.head(h)
        # split the 3*K outputs into weights, means, and sigmas
        pi_logits = out[:, :self.K]
        mu = out[:, self.K:2*self.K]
        log_sigma = out[:, 2*self.K:]
        # softplus ensures sigma > 0, floor prevents collapse
        sigma = F.softplus(log_sigma) + self.sigma_floor
        return pi_logits, mu, sigma


class DiscreteBins(nn.Module):
    """Softmax over discrete price bins. Predicts P(payprice == k).
    Simpler than MDN -- just output probability for each integer price.
    """

    def __init__(self, vocab_sizes, emb_dims, tag_vocab_size, tag_emb_dim,
                 num_continuous, hidden, dropout, num_bins):
        super().__init__()
        self.emb = CategoricalEmbeddings(vocab_sizes, emb_dims, tag_vocab_size, tag_emb_dim)
        self.cont_norm = nn.Identity()
        in_dim = self.emb.out_dim + num_continuous
        self.backbone = MLPBackbone(in_dim, hidden, dropout)
        self.num_bins = num_bins
        self.head = nn.Linear(self.backbone.out_dim, num_bins)

    def forward(self, cat, cont, tags):
        e = self.emb(cat, tags)
        c = self.cont_norm(cont)
        x = torch.cat([e, c], dim=1)
        h = self.backbone(x)
        # returns raw logits, apply softmax externally for probabilities
        return self.head(h)


def build_vocab_sizes(artifacts):
    """Get vocab size per feature from the saved artifacts.
    +1 because index 0 is reserved for unknown/rare values.
    """
    vs = {}
    for feat in artifacts['categorical_features']:
        vs[feat] = len(artifacts['encoders'][feat]) + 1
    return vs
