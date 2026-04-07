import torch
import torch.nn as nn
import pickle


EMBEDDING_DIM = 16
HIDDEN_DIM = 128
FFN_DIM = 256
NUM_HEADS = 4
NUM_LAYERS = 3
DROPOUT = 0.1

# number of continuous + cyclical features (1 continuous + 4 cyclical)
N_CONTINUOUS = 5


def build_embedding_layers(artifacts):
    # build one embedding layer per categorical feature
    encoders = artifacts["encoders"]
    embeddings = {}
    for col, encoder in encoders.items():
        num_categories = len(encoder.classes_)
        embeddings[col] = nn.Embedding(num_categories, EMBEDDING_DIM)
    return nn.ModuleDict(embeddings)


class BidTransformer(nn.Module):
    def __init__(self, artifacts):
        super().__init__()

        encoders = artifacts["encoders"]
        tag_vocab = artifacts["tag_vocab"]

        # one embedding per categorical feature
        self.cat_embeddings = build_embedding_layers(artifacts)
        self.cat_feature_order = list(encoders.keys())

        # tag embedding -- each tag ID maps to a vector, then I mean pool them
        tag_vocab_size = len(tag_vocab) + 1  # +1 for padding index 0
        self.tag_embedding = nn.Embedding(tag_vocab_size, EMBEDDING_DIM, padding_idx=0)

        # total input dim after concatenating all embeddings + continuous features
        n_cat = len(encoders)
        input_dim = n_cat * EMBEDDING_DIM + EMBEDDING_DIM + N_CONTINUOUS

        # project the concatenated input vector to hidden dim for the transformer
        self.input_projection = nn.Linear(input_dim, HIDDEN_DIM)

        # transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=HIDDEN_DIM,
            nhead=NUM_HEADS,
            dim_feedforward=FFN_DIM,
            dropout=DROPOUT,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=NUM_LAYERS)

        # output head -- predicts mu and sigma of the log-normal distribution
        self.output_head = nn.Linear(HIDDEN_DIM, 2)

        # softplus keeps sigma always positive
        self.softplus = nn.Softplus()

        # initialize output head with small weights so mu/sigma don't explode right at the start
        # without this sigma can collapse near zero immediately and NLL goes crazy
        nn.init.xavier_uniform_(self.output_head.weight, gain=0.01)
        nn.init.zeros_(self.output_head.bias)

    def forward(self, cats, conts, tags):
        # cats: [batch, n_categorical] int64
        # conts: [batch, n_continuous] float32
        # tags: [batch, max_tag_seq_len] int64, padded with 0

        # embed each categorical feature and concatenate them all
        cat_embeds = []
        for i, col in enumerate(self.cat_feature_order):
            embed = self.cat_embeddings[col](cats[:, i])
            cat_embeds.append(embed)
        cat_embeds = torch.cat(cat_embeds, dim=1)  # [batch, n_cat * emb_dim]

        # embed tags and mean pool -- ignore padding index 0 in the mean
        tag_embeds = self.tag_embedding(tags)       # [batch, seq_len, emb_dim]
        mask = (tags != 0).float().unsqueeze(-1)    # [batch, seq_len, 1]
        tag_sum = (tag_embeds * mask).sum(dim=1)    # [batch, emb_dim]
        tag_count = mask.sum(dim=1).clamp(min=1)    # avoid division by zero
        tag_mean = tag_sum / tag_count              # [batch, emb_dim]

        # concatenate everything into one feature vector
        x = torch.cat([cat_embeds, tag_mean, conts], dim=1)  # [batch, input_dim]

        # project to hidden dim
        x = self.input_projection(x)   # [batch, hidden_dim]

        # transformer expects [batch, seq_len, hidden_dim], seq_len=1 for tabular data
        x = x.unsqueeze(1)             # [batch, 1, hidden_dim]
        x = self.transformer(x)        # [batch, 1, hidden_dim]
        x = x.squeeze(1)              # [batch, hidden_dim]

        # output head
        out = self.output_head(x)          # [batch, 2]
        mu = out[:, 0]                     # [batch]
        sigma = self.softplus(out[:, 1])   # [batch], always > 0

        return mu, sigma


class MLPBaseline(nn.Module):
    def __init__(self, artifacts):
        super().__init__()

        encoders = artifacts["encoders"]
        tag_vocab = artifacts["tag_vocab"]

        # same embedding setup as BidTransformer so the comparison is fair
        self.cat_embeddings = build_embedding_layers(artifacts)
        self.cat_feature_order = list(encoders.keys())

        tag_vocab_size = len(tag_vocab) + 1
        self.tag_embedding = nn.Embedding(tag_vocab_size, EMBEDDING_DIM, padding_idx=0)

        n_cat = len(encoders)
        input_dim = n_cat * EMBEDDING_DIM + EMBEDDING_DIM + N_CONTINUOUS

        # 3 hidden layers with batch norm and ReLU
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, cats, conts, tags):
        # same input construction as BidTransformer
        cat_embeds = []
        for i, col in enumerate(self.cat_feature_order):
            embed = self.cat_embeddings[col](cats[:, i])
            cat_embeds.append(embed)
        cat_embeds = torch.cat(cat_embeds, dim=1)

        tag_embeds = self.tag_embedding(tags)
        mask = (tags != 0).float().unsqueeze(-1)
        tag_sum = (tag_embeds * mask).sum(dim=1)
        tag_count = mask.sum(dim=1).clamp(min=1)
        tag_mean = tag_sum / tag_count

        x = torch.cat([cat_embeds, tag_mean, conts], dim=1)

        out = self.layers(x)
        return out.squeeze(1)  # [batch]


def load_artifacts(path="data/processed/artifacts.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
