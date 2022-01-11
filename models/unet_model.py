""" Full assembly of the parts to form the complete network """

from .unet_parts import *

# generative multi-column convolutional neural net
class DMFB(nn.Module):
    def __init__(self):
        super(DMFB, self).__init__()
        self.conv_3 = nn.Conv2d(512, 128, 3, 1, 1)
        conv_3_sets = []
        for i in range(4):
            conv_3_sets.append(nn.Conv2d(128, 128, 3, padding=1))
        self.conv_3_sets = nn.ModuleList(conv_3_sets)
        self.conv_3_2 = nn.Conv2d(128, 128, 3, padding=2, dilation=2)
        self.conv_3_4 = nn.Conv2d(128, 128, 3, padding=4, dilation=4)
        self.conv_3_8 = nn.Conv2d(128, 128, 3, padding=8, dilation=8)
        self.act_fn = nn.Sequential(nn.ReLU(), nn.InstanceNorm2d(512))
        self.conv_1 = nn.Conv2d(512, 512, 1)
        self.norm = nn.InstanceNorm2d(512)

    def forward(self, inputs):
        src = inputs
        # conv-3
        x = self.act_fn(inputs)
        x = self.conv_3(x)

        K = []
        for i in range(4):
            if i != 0:
                p = eval('self.conv_3_' + str(2 ** i))(x)
                p = p + K[i - 1]
            else:
                p = x
            K.append(self.conv_3_sets[i](p))
        cat = torch.cat(K, 1)
        bottle = self.conv_1(self.norm(cat))
        out = bottle + src
        return out

class UNet(nn.Module):
    def __init__(self, n_channels, out_channels, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = out_channels
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        #self.dmfb_ = nn.Sequential(*[DMFB() for _ in range(1)])
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, out_channels)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        #x5 = self.dmfb_(x5) + x5
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
