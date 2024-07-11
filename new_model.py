import torch
import torch.nn as nn

class ConvBlock3D(nn.Module):
    '''(Conv3d => BN => ReLU) x 2'''
    def __init__(self, in_ch, out_ch):
        super(ConvBlock3D, self).__init__()
        self.convblock = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.convblock(x)
    

class DownBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DownBlock3D, self).__init__()
        self.convblock = ConvBlock3D(in_ch, out_ch)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        x_a =  self.convblock(x)
        x_b = self.pool(x_a)
        return x_a, x_b


class UpBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UpBlock3D, self).__init__()
        self.upsample = nn.ConvTranspose3d(in_ch, in_ch, kernel_size=2, stride=2, padding=0)
        self.convblock = ConvBlock3D(2 * in_ch, out_ch)

    def forward(self, x, x_new):
        x = self.upsample(x)
        # When using Conv3D, ensure the dimensions match for concatenation
        # Padding or slicing may be required here if dimensions don't match exactly
        x = torch.cat([x, x_new], dim=1)
        x = self.convblock(x)
        return x


class UNet3D(nn.Module):
    def __init__(self):
        super(UNet3D, self).__init__()

        self.base = 32

        # Define the network layers using the base size
        self.down0 = DownBlock3D(in_ch=1, out_ch=self.base)
        self.down1 = DownBlock3D(in_ch=self.base, out_ch=2*self.base)
        self.down2 = DownBlock3D(in_ch=2*self.base, out_ch=4*self.base)
        self.down3 = DownBlock3D(in_ch=4*self.base, out_ch=8*self.base)
        self.conv = ConvBlock3D(in_ch=8*self.base, out_ch=8*self.base)
        self.up4 = UpBlock3D(in_ch=8*self.base, out_ch=4*self.base)
        self.up3 = UpBlock3D(in_ch=4*self.base, out_ch=2*self.base)
        self.up2 = UpBlock3D(in_ch=2*self.base, out_ch=self.base)
        self.up1 = UpBlock3D(in_ch=self.base, out_ch=self.base)
        self.outconv = nn.Conv3d(self.base, 1, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        x0_a, x0_b = self.down0(x)
        x1_a, x1_b = self.down1(x0_b)
        x2_a, x2_b = self.down2(x1_b)
        x3_a, x3_b = self.down3(x2_b)
        x4 = self.conv(x3_b)
        x3 = self.up4(x4, x3_a)
        x2 = self.up3(x3, x2_a)
        x1 = self.up2(x2, x1_a)
        x0 = self.up1(x1, x0_a)
        x = self.outconv(x0)
        return x
