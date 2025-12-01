import torch
import torch.nn as nn
import torch.nn.functional as F

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_se=True):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.use_se = use_se
        if self.use_se: self.se = SELayer(out_channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        if self.use_se: out = self.se(out)
        out += residual
        return F.relu(out, inplace=True)

class ChessNet(nn.Module):
    def __init__(self, num_res_blocks=10, num_channels=128, use_se=True):
        super(ChessNet, self).__init__()
        
        # 1. Tronc Commun (Backbone)
        self.conv_input = nn.Sequential(
            nn.Conv2d(17, num_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(inplace=True)
        )
        
        blocks = [ResidualBlock(num_channels, num_channels, use_se) for _ in range(num_res_blocks)]
        self.res_tower = nn.Sequential(*blocks)
        
        # 2. TÊTE 1 : POLICY (Quel coup jouer ?)
        self.policy_head = nn.Sequential(
            nn.Conv2d(num_channels, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 4096)
        )
        
        # 3. TÊTE 2 : VALUE (Qui va gagner ?)
        # Sortie : Une seule valeur entre -1 (Perdu) et +1 (Gagné)
        self.value_head = nn.Sequential(
            nn.Conv2d(num_channels, 1, kernel_size=1), # Réduit à 1 canal
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
            nn.Tanh() # Tanh force la sortie entre -1 et 1
        )

    def forward(self, x):
        # Passage dans le tronc commun
        x = self.conv_input(x)
        x = self.res_tower(x)
        
        # Bifurcation vers les deux têtes
        policy = self.policy_head(x)
        value = self.value_head(x)
        
        return policy, value