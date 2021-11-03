import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, z_dim=100, input_c=1):
        super(VAE, self).__init__()

        self.z_dim = z_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_c, 32, kernel_size=4, stride=2, padding=1), # 128 -> 64
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1),    # 64 -> 32 
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),    # 32 -> 32
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),    # 32 -> 16
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),    # 16 -> 16
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),   # 16 -> 8
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),   # 8 -> 8
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),    # 8 -> 8
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(32, z_dim, kernel_size=8, stride=1))            # 8 -> 1
        
        self.mu_fc = nn.Linear(z_dim, z_dim)
        self.logvar_fc = nn.Linear(z_dim, z_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=8, mode='nearest'),
            nn.Conv2d(z_dim, 32, kernel_size=3, stride=1, padding=1),  # 1 -> 8
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),     # 8 -> 8
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),    # 8 -> 8
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),    # 8 -> 16
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),     # 16 -> 16
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),     # 16 -> 32
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),     # 32 -> 32
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),     # 32 -> 64
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(32, input_c, kernel_size=3, stride=1, padding=1),  # 64 -> 128
            nn.Sigmoid())

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
        
    def reparameterize(self, mu, logvar):
        #std = logvar.mul(0.5).exp_()
        #eps = std.new(std.size()).normal_()
        #return eps.mul(std).add_(mu)
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        h      = self.encoder(x)
        
        h = torch.flatten(h, start_dim=1)
        mu     = self.mu_fc(h)     # 平均ベクトル
        logvar = self.logvar_fc(h) # 分散共分散行列の対数
        z      = self.reparameterize(mu, logvar)  # 潜在変数

        x_hat  = self.decoder(z.view(z.size(0), -1, 1, 1))
        self.mu     = mu.squeeze()
        self.logvar = logvar.squeeze()
        return x_hat
    

class AE(nn.Module):
    def __init__(self, z_dim=100, input_c=1):
        super(AE, self).__init__()

        self.z_dim = z_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_c, 32, kernel_size=4, stride=2, padding=1), # 128 -> 64
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1),    # 64 -> 32 
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),    # 32 -> 32
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),    # 32 -> 16
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),    # 16 -> 16
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),   # 16 -> 8
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),   # 8 -> 8
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),    # 8 -> 8
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(32, z_dim, kernel_size=8, stride=1))            # 8 -> 1

        # Decoder
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=8, mode='nearest'),
            nn.Conv2d(z_dim, 32, kernel_size=3, stride=1, padding=1),  # 1 -> 8
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),     # 8 -> 8
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),    # 8 -> 8
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),    # 8 -> 16
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),     # 16 -> 16
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),     # 16 -> 32
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),     # 32 -> 32
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),     # 32 -> 64
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(32, input_c, kernel_size=3, stride=1, padding=1),  # 64 -> 128
            nn.Sigmoid())

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        h      = self.encoder(x)
        x_hat  = self.decoder(h)
        return x_hat
