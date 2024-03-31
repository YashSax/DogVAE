import torch
import torch.nn as nn

class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dim):
        super(VariationalAutoencoder, self).__init__()

        in_channels =  [3,  32,  64, 128, 128]
        out_channels = [32, 64, 128, 128, 128]
        self.latent_dim = latent_dim

        modules = []
        for in_channel, out_channel in zip(in_channels, out_channels):
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channel, out_channel, 5, 2, 1),
                    nn.BatchNorm2d(out_channel),
                    nn.ReLU()
                )
            )
        
        self.encoder = nn.Sequential(*modules)
        encoder_out = 1152

        self.final_encoder_layer = nn.Linear(encoder_out, self.latent_dim)

        self.fc_mu = nn.Linear(self.latent_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(self.latent_dim, self.latent_dim)

        self.first_decoder_layer = nn.Linear(self.latent_dim, encoder_out)

        modules = []
        in_channels, out_channels = out_channels[::-1], in_channels[::-1]

        for in_channel, out_channel in zip(in_channels, out_channels):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_channel, out_channel, 5, 2, 1),
                    nn.BatchNorm2d(out_channel),
                    nn.ReLU()
                )
            )
        self.decoder = nn.Sequential(*modules)
        self.final_decoder_layer = nn.Sequential(
            nn.Conv2d(3, 3, 5, 1, 2),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.Conv2d(3, 3, 5, 1, 2)
        )

    def encode(self, x):
        x = self.encoder(x)
        x = x.flatten(start_dim=1)
        x = self.final_encoder_layer(x)
        return x

    def reparametrize(self, x):
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        std = torch.sqrt(torch.exp(logvar))
        eps = torch.rand_like(std)

        return mu, logvar, mu + std * eps
    
    def decode(self, z):
        z = self.first_decoder_layer(z)
        z = z.view(-1, 128, 3, 3)
        z = self.decoder(z)
        z = self.final_decoder_layer(z)
        return z

    def forward(self, x):
        x = self.encode(x)

        mu, logvar, z = self.reparametrize(x)

        return self.decode(z), mu, logvar
