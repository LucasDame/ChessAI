import torch
import torch.nn as nn
import torch.nn.functional as F

class ChessNet(nn.Module):
    def __init__(self):
        super(ChessNet, self).__init__()
        
        # 17 plans d'entrée :
        # 0-5: Pièces blanches, 6-11: Pièces noires
        # 12: Trait, 13-16: Droits de roque
        self.conv1 = nn.Conv2d(17, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        
        # Couches linéaires
        # 8*8*256 = 16384 caractéristiques extraites
        self.fc1 = nn.Linear(256 * 8 * 8, 1024)
        self.dropout = nn.Dropout(0.3) # Pour éviter le surapprentissage
        self.fc2 = nn.Linear(1024, 4096) # Sortie : 4096 coups possibles

    def forward(self, x):
        # x shape: (batch, 17, 8, 8)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        x = x.view(-1, 256 * 8 * 8) # Aplatir
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x