import torch
import torch.nn as nn
import torch.nn.functional as F

class SELayer(nn.Module):
    """
    Squeeze-and-Excitation Block (SE).
    Permet au réseau de recalibrer l'importance des canaux (features).
    """
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        # Squeeze: Global Average Pooling (réduit 8x8 en 1x1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # Excitation: Petit réseau dense pour apprendre les poids
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
        # On multiplie les canaux d'origine par les poids calculés
        return x * y.expand_as(x)

class ResidualBlock(nn.Module):
    """
    Bloc Résiduel standard avec option SE.
    Structure : Input -> Conv -> BN -> ReLU -> Conv -> BN -> (SE) -> Add -> ReLU
    """
    def __init__(self, in_channels, out_channels, use_se=False):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.use_se = use_se
        if self.use_se:
            self.se = SELayer(out_channels)

    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Option Squeeze-and-Excitation
        if self.use_se:
            out = self.se(out)
            
        # Connexion Résiduelle (Skip Connection)
        out += residual
        out = F.relu(out, inplace=True)
        return out

class ChessNet(nn.Module):
    """
    Réseau Profond (ResNet) pour les échecs.
    """
    def __init__(self, num_res_blocks=10, num_channels=128, use_se=True):
        super(ChessNet, self).__init__()
        
        # --- 1. Tête d'Entrée (Input Head) ---
        # 17 plans en entrée -> monte à num_channels (ex: 128 ou 256)
        self.conv_input = nn.Sequential(
            nn.Conv2d(17, num_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(inplace=True)
        )
        
        # --- 2. Tour Résiduelle (Backbone) ---
        # Empilement de N blocs résiduels
        blocks = []
        for _ in range(num_res_blocks):
            blocks.append(ResidualBlock(num_channels, num_channels, use_se=use_se))
        self.res_tower = nn.Sequential(*blocks)
        
        # --- 3. Tête de Politique (Policy Head) ---
        # Réduit les canaux pour préparer la sortie finale
        self.policy_conv = nn.Sequential(
            nn.Conv2d(num_channels, 32, kernel_size=1), # Convolution 1x1 pour réduire la profondeur
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # Couche entièrement connectée pour les 4096 coups
        # 32 canaux * 8 * 8 cases = 2048 neurones en entrée du linéaire
        self.policy_fc = nn.Linear(32 * 8 * 8, 4096)

    def forward(self, x):
        # 1. Input
        x = self.conv_input(x)
        
        # 2. ResNet Tower
        x = self.res_tower(x)
        
        # 3. Policy Head
        x = self.policy_conv(x)
        x = x.view(x.size(0), -1) # Aplatir (Flatten)
        x = self.policy_fc(x)
        
        return x