import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable


class Encoder_Block(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, dropout=0.):
        super(Encoder_Block, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=mid_channels,
                               kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(mid_channels)

        self.conv2 = nn.Conv2d(in_channels=mid_channels, out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.dropout(x)

        return x

    def __call__(self, x):
        return self.forward(x)

class Decoder_Block(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, dropout=0.):
        super(Decoder_Block, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=mid_channels,
                               kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(mid_channels)

        self.conv2 = nn.Conv2d(in_channels=mid_channels, out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.dropout(x)

        return x

    def __call__(self, x):
        return self.forward(x)

class Res_Bottleneck(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, dropout=0.):
        super(Res_Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=mid_channels,
                               kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(mid_channels)

        self.conv2 = nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels,
                               kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(mid_channels)

        self.res_conv1 = nn.Conv2d(in_channels=in_channels, out_channels=mid_channels,
                               kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn1_res = nn.BatchNorm2d(mid_channels)

        self.conv3 = nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels,
                               kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn3 = nn.BatchNorm2d(mid_channels)

        self.conv4 = nn.Conv2d(in_channels=mid_channels, out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn4 = nn.BatchNorm2d(out_channels)

        self.res_conv2 = nn.Conv2d(in_channels=mid_channels, out_channels=out_channels,
                                   kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn2_res = nn.BatchNorm2d(out_channels)

        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x):
        shortcut = self.res_conv1(x)
        shortcut = self.bn1_res(shortcut)


        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = x+shortcut
        x = F.relu(x)

        shortcut = self.res_conv2(x)
        shortcut = self.bn2_res(shortcut)


        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = x+shortcut
        x = F.relu(x)

        x = self.dropout(x)

        return x


class Mask_Net(nn.Module):

    def __init__(self, in_channel, out_channel):
        super(Mask_Net, self).__init__()
        self.input_adapt = nn.Sequential(
            nn.Conv2d(kernel_size=(7, 7), in_channels=in_channel, out_channels=32, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.conv_encode1 = Encoder_Block(in_channels=32, mid_channels=32, out_channels=32)
        self.conv_avgpool1 = nn.AvgPool2d(kernel_size=2)

        self.conv_encode2 = Encoder_Block(in_channels=32, mid_channels=64, out_channels=64)
        self.conv_avgpool2 = nn.AvgPool2d(kernel_size=2)

        self.conv_encode3 = Encoder_Block(in_channels=64, mid_channels=128, out_channels=128)
        self.conv_avgpool3 = nn.AvgPool2d(kernel_size=2)

        self.conv_encode4 = nn.Sequential(
            nn.Conv2d(kernel_size=(3, 3), in_channels=128, out_channels=256, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(0.3)
        )
        self.conv_avgpool4 = nn.AvgPool2d(kernel_size=2)
        # 瓶颈层
        self.bottleneck = Res_Bottleneck(in_channels=256, mid_channels=512, out_channels=512, dropout=0.5)

        # 语义差异
        self.link_4 = nn.Sequential(
            nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

        )
        self.link_3 = nn.Sequential(
            nn.Conv2d(kernel_size=(3, 3), in_channels=128, out_channels=128, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(kernel_size=(3, 3), in_channels=128, out_channels=128, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.up_sampling4 = nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=(3, 3), stride=(2, 2),
                                     padding=(1, 1), output_padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.conv_decode4 = nn.Sequential(
            nn.Conv2d(kernel_size=(3, 3), in_channels=512, out_channels=256, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(0.3)
        )

        self.up_sampling3 = nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=(3, 3), stride=(2, 2),
                                     padding=(1, 1), output_padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.conv_decode3 = Decoder_Block(in_channels=256, mid_channels=128, out_channels=64)

        self.up_sampling2 = nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(2, 2),
                                     padding=(1, 1), output_padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.conv_decode2 = Decoder_Block(in_channels=64, mid_channels=64, out_channels=32)

        self.up_sampling1 = nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(2, 2),
                                     padding=(1, 1), output_padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.conv_decode1 = Decoder_Block(in_channels=32, mid_channels=32, out_channels=32)

        self.output_layer = nn.Sequential(
            nn.Conv2d(kernel_size=(3, 3), in_channels=32, out_channels=out_channel, padding=1),
            # nn.BatchNorm2d(out_channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Encode
        x = self.input_adapt(x)

        x = self.conv_encode1(x)
        x = self.conv_avgpool1(x)

        x = self.conv_encode2(x)
        x = self.conv_avgpool2(x)

        encode3 = self.conv_encode3(x)
        x = self.conv_avgpool3(encode3)

        encode4 = self.conv_encode4(x)
        x = self.conv_avgpool4(encode4)

        x = self.bottleneck(x)

        x = self.up_sampling4(x)
        encode4 = self.link_4(encode4)
        x = torch.concat((encode4, x), dim=1)
        x = self.conv_decode4(x)

        x = self.up_sampling3(x)
        encode3 = self.link_3(x)
        x = torch.concat((encode3, x), dim=1)
        x = self.conv_decode3(x)

        x = self.up_sampling2(x)
        x = self.conv_decode2(x)

        x = self.up_sampling1(x)
        x = self.conv_decode1(x)

        final_layer = self.output_layer(x)
        return final_layer


class Asymmetry_Encoder_Block(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, dropout=0.):
        super(Asymmetry_Encoder_Block, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=mid_channels,
                               kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(mid_channels)

        self.conv2 = nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels,
                               kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(mid_channels)

        self.res_conv1 = nn.Conv2d(in_channels=in_channels, out_channels=mid_channels,
                                   kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn1_res = nn.BatchNorm2d(mid_channels)

        self.conv3 = nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels,
                               kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn3 = nn.BatchNorm2d(mid_channels)

        self.conv4 = nn.Conv2d(in_channels=mid_channels, out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn4 = nn.BatchNorm2d(out_channels)

        self.res_conv2 = nn.Conv2d(in_channels=mid_channels, out_channels=out_channels,
                                   kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn2_res = nn.BatchNorm2d(out_channels)

        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x):
        shortcut = self.res_conv1(x)
        shortcut = self.bn1_res(shortcut)

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = x + shortcut
        x = F.relu(x)

        shortcut = self.res_conv2(x)
        shortcut = self.bn2_res(shortcut)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = x + shortcut
        x = F.relu(x)

        x = self.dropout(x)

        return x

class Asymmetry_Decoder_Block(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, dropout):
        super(Asymmetry_Decoder_Block, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=mid_channels,
                               kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(mid_channels)

        self.conv2 = nn.Conv2d(in_channels=mid_channels, out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.dropout(x)

        return x

    def __call__(self, x):
        return self.forward(x)


class Asymmetry_Net(nn.Module):

    def __init__(self, in_channel):
        super(Asymmetry_Net, self).__init__()
        self.input_adapt = nn.Sequential(
            nn.Conv2d(kernel_size=(7, 7), in_channels=in_channel, out_channels=32, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.conv_encode1 = Asymmetry_Encoder_Block(in_channels=32, mid_channels=32, out_channels=32)
        self.conv_avgpool1 = nn.AvgPool2d(kernel_size=2)

        self.conv_encode2 = Asymmetry_Encoder_Block(in_channels=32, mid_channels=64, out_channels=64)
        self.conv_avgpool2 = nn.AvgPool2d(kernel_size=2)

        self.conv_encode3 = Asymmetry_Encoder_Block(in_channels=64, mid_channels=128, out_channels=128)
        self.conv_avgpool3 = nn.AvgPool2d(kernel_size=2)

        self.conv_encode4 = Asymmetry_Encoder_Block(in_channels=128, mid_channels=256, out_channels=256, dropout=0.3)
        self.conv_avgpool4 = nn.AvgPool2d(kernel_size=2)
        # 瓶颈层
        self.bottleneck = Res_Bottleneck(in_channels=256, mid_channels=512, out_channels=512, dropout=0.5)

        self.up_sampling4 = nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=(3, 3), stride=(2, 2),
                                     padding=(1, 1), output_padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(0.3)
        )
        self.conv_decode4 = nn.Sequential(
            nn.Conv2d(kernel_size=(3, 3), in_channels=512, out_channels=256, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(kernel_size=(3, 3), in_channels=256, out_channels=256, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.up_sampling3 = nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=(3, 3), stride=(2, 2),
                                     padding=(1, 1), output_padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.conv_decode3 = Decoder_Block(in_channels=256, mid_channels=128, out_channels=128)

        self.up_sampling2 = nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(3, 3), stride=(2, 2),
                                     padding=(1, 1), output_padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.conv_decode2 = Decoder_Block(in_channels=128, mid_channels=64, out_channels=64)

        self.up_sampling1 = nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(3, 3), stride=(2, 2),
                                     padding=(1, 1), output_padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.conv_decode1 = Decoder_Block(in_channels=64, mid_channels=32, out_channels=32)

        self.output_layer = nn.Sequential(
            nn.Conv2d(kernel_size=(3, 3), in_channels=32, out_channels=2, padding=1),
            nn.BatchNorm2d(2),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Encode
        x = self.input_adapt(x)

        encode1 = self.conv_encode1(x)
        x = self.conv_avgpool1(encode1)

        encode2 = self.conv_encode2(x)
        x = self.conv_avgpool2(encode2)

        encode3 = self.conv_encode3(x)
        x = self.conv_avgpool3(encode3)

        encode4 = self.conv_encode4(x)
        x = self.conv_avgpool4(encode4)

        x = self.bottleneck(x)

        x = self.up_sampling4(x)
        x = torch.concat((encode4, x), dim=1)
        x = self.conv_decode4(x)

        x = self.up_sampling3(x)
        x = torch.concat((encode3, x), dim=1)
        x = self.conv_decode3(x)

        x = self.up_sampling2(x)
        x = torch.concat((encode2, x), dim=1)
        x = self.conv_decode2(x)

        x = self.up_sampling1(x)
        x = torch.concat((encode1, x), dim=1)
        x = self.conv_decode1(x)

        final_layer = self.output_layer(x)
        return final_layer

