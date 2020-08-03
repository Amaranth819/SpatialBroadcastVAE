import torch
import torch.nn as nn

'''
    Spatial Broadcast VAE
'''
class SBVAE(nn.Module):
    def __init__(self, D = 64, in_c = 3):
        super(SBVAE, self).__init__()
        self.D = D
        self.in_c = in_c

        # Encoder
        encoder_out_D = D // 2**4
        self.conv_encoder = nn.Sequential(
            nn.Conv2d(in_c, 32, kernel_size = 4, stride = 2, padding = 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(32, 32, kernel_size = 4, stride = 2, padding = 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(32, 64, kernel_size = 4, stride = 2, padding = 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(64, 64, kernel_size = 4, stride = 2, padding = 1),
            nn.ReLU(inplace = True),
            nn.Flatten(),
            nn.Linear(encoder_out_D * encoder_out_D * 64, 256),
            nn.ReLU(inplace = True),
            nn.Linear(256, 20)
        )

        # Coordinates
        x_ls = torch.linspace(-1, 1, D)
        y_ls = torch.linspace(-1, 1, D)
        self.x_axis, self.y_axis = torch.meshgrid(x_ls, y_ls)
        self.x_axis = self.x_axis.view(1, 1, *self.x_axis.size())
        self.y_axis = self.y_axis.view(1, 1, *self.y_axis.size())

        # Decoder
        self.broadcast_decoder = nn.Sequential(
            nn.Conv2d(12, 64, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(64, 64, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(64, in_c, kernel_size = 3, stride = 1, padding = 1),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(logvar * 0.5)
        eps = torch.randn_like(std).to(std.get_device())
        return mu + eps * std

    def loss_func(self, x, recon_x, mu, logvar):
        bs = x.size(0)
        mse_loss = torch.mean(torch.sum(nn.MSELoss(reduction = 'none')(recon_x, x), dim = [1, 2, 3]))
        kl_div = 0.5 * torch.mean(torch.sum(logvar.exp() + mu.pow(2) - 1 - logvar, dim = 1))
        return mse_loss, kl_div

    def forward(self, x):
        # Encoder
        x = self.conv_encoder(x)
        mu, logvar = torch.split(x, 10, dim = 1)

        # Sampling
        z = self.reparameterize(mu, logvar)
        device = z.get_device()

        # Append coordinate channels
        bs = z.size(0)
        X, Y = self.x_axis.expand(bs, -1, -1, -1).to(device), self.y_axis.expand(bs, -1, -1, -1).to(device)
        z = z.view(*z.size(), 1, 1).expand(-1, -1, self.D, self.D)
        z = torch.cat((X, Y, z), dim = 1)

        # Decoder
        out = self.broadcast_decoder(z)

        return mu, logvar, out

def truncated_normal_initializer(tensor, mean = 0, std = 0.1):
    # https://zhuanlan.zhihu.com/p/83609874
    with torch.no_grad():
        size = tensor.size()
        tmp = tensor.new_empty(size + (4, )).normal_()
        valid = (tmp < 2) & (tmp > -2)
        ind = valid.max(-1, keepdim = True)[1]
        tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
        tensor.data.mul_(std).add_(mean)
        return tensor

def weight_init(net):
    for module in net.modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            module.weight.data = truncated_normal_initializer(module.weight.data)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

if __name__ == '__main__':
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    sv = SBVAE().to(device)
    x = torch.zeros([16, 3, 64, 64]).to(device)
    mu, logvar, y = sv(x)
    print(y.size())
    mse, kl = sv.loss_func(x, y, mu, logvar)
    print(mse.size(), kl.size())
