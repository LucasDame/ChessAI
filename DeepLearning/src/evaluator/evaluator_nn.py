import torch
import torch.nn as nn

class Evaluator(nn.Module):

    def __init__(self, model_path):
        super(Evaluator, self).__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(17 * 8 * 8, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, board):
        return self.net(board)
    
def load_evaluator(model_path):
    model = Evaluator(model_path)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model