import torch
import torch.nn as nn

FEATURE_DIM = 6  # must match len(scraper_agent.features_to_vector(...))


class LeadScoringModel(nn.Module):
    """Small, explainable MLP — deliberately simple. With 15-20 real labeled
    samples, a bigger model would just memorize the training set. Scale this
    up only once you have hundreds of labeled examples.
    """
    def __init__(self, input_dim: int = FEATURE_DIM, hidden_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


def load_model(path: str) -> LeadScoringModel:
    model = LeadScoringModel()
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model
