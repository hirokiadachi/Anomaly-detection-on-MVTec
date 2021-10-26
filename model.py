import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, z_dim=100, input_c=1):
        super(VAE, self).__init__()

        self.z_dim = z_dim
        num_layers = 5
        
        # encode
        in_ch = input_c
        out_ch = 16
        encoder = []
        for _ in range(num_layers):
            encoder.append(nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False))
            encoder.append(nn.BatchNorm2d(out_ch))
            encoder.append(nn.ReLU(inplace=True))
            in_ch = out_ch
            out_ch = out_ch*2
        
        self.mu_fc = nn.Linear(in_ch*4**2, z_dim)
        self.logvar_fc = nn.Linear(in_ch*4**2, z_dim)
        self.decoder_fc = nn.Linear(z_dim, in_ch*4**2)

        in_ch = in_ch
        out_ch = 64
        decoder = []
        for _ in range(num_layers - 1):
            decoder.append(nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=True))
            decoder.append(nn.BatchNorm2d(out_ch))
            decoder.append(nn.ReLU(inplace=True))
            in_ch = out_ch
            out_ch = out_ch*2
        decoder.append(nn.ConvTranspose2d(in_ch, input_c, kernel_size=4, stride=2, padding=1))
        decoder.append(nn.Sigmoid())
        
        self.encoder = nn.Sequential(*encoder)
        self.decoder = nn.Sequential(*decoder)

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
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x):
        h      = self.encoder(x)
        
        h = torch.flatten(h, start_dim=1)
        mu     = self.mu_fc(h)     # 平均ベクトル
        logvar = self.logvar_fc(h) # 分散共分散行列の対数
        z      = self.reparameterize(mu, logvar)  # 潜在変数

        x_hat  = self.decoder(self.decoder_fc(z).view(z.size(0), -1, 4, 4))
        self.mu     = mu.squeeze()
        self.logvar = logvar.squeeze()
        return x_hat