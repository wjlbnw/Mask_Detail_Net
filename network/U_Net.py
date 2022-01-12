import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable


class Unet(nn.Module):

    def contracting_block(self, in_channels, out_channels, kernel_size=3, padding=1):
        block = nn.Sequential(
            nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=out_channels, padding=padding),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),  # 数据归一化
            nn.Conv2d(kernel_size=kernel_size, in_channels=out_channels, out_channels=out_channels, padding=padding),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),  # 数据归一化
        )
        return block

    def expansive_block(self, in_channels, mid_channel, out_channels, kernel_size=3, padding=1):
        block = torch.nn.Sequential(
            torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel, padding=padding),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(mid_channel),
            torch.nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=mid_channel, padding=padding),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(mid_channel),
            # 反卷积，输出的尺寸为(inputsize-1)*stride-2padding+k+output_padding和卷积公式算算就出来了
            torch.nn.ConvTranspose2d(in_channels=mid_channel, out_channels=out_channels, kernel_size=3, stride=2,
                                     padding=1, output_padding=1),
        )
        return block
    def final_block(self, in_channels, mid_channel, out_channels, kernel_size=3, padding=1):
        block = torch.nn.Sequential(
            torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel, padding=padding),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(mid_channel),
            torch.nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=mid_channel, padding=padding),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(mid_channel),
            torch.nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=out_channels, padding=padding),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(out_channels),
        )
        return block
    def __init__(self, in_channel, out_channel):
        super(Unet, self).__init__()
        self.conv_encode1 = self.contracting_block(in_channels=in_channel, out_channels=64)
        self.conv_maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv_encode2 = self.contracting_block(64, 128, padding=1)
        self.conv_maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.conv_encode3 = self.contracting_block(128, 256, padding=1)
        self.conv_maxpool3 = nn.MaxPool2d(kernel_size=2)
        self.conv_encode4 = self.contracting_block(256, 512, padding=1)
        self.conv_maxpool4 = nn.MaxPool2d(kernel_size=2)
        # 图中最底下一层
        self.bottleneck = nn.Sequential(
            nn.Conv2d(kernel_size=3, in_channels=512, out_channels=1024, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(1024),  # 数据归一化
            nn.Conv2d(kernel_size=3, in_channels=1024, out_channels=1024, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(1024),  # 数据归一化
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=3, stride=2,
                                     padding=1, output_padding=1)
        )
        self.conv_decode4 = self.expansive_block(1024, 512, 256)
        self.conv_decode3 = self.expansive_block(512, 256, 128)
        self.conv_decode2 = self.expansive_block(256, 128, 64)
        self.final_layer = self.final_block(128, 64, out_channel)
    def crop_and_concat(self, upsampled, bypass, crop=False):
        '''
        拼接，联系在一起
        :param upsampled:
        :param bypass:
        :param crop:
        :return:
        '''

        if crop:
            c = (bypass.size()[2] - upsampled.size()[2]) // 2
            bypass = F.pad(bypass, (-c, -c, -c, -c))
        return torch.cat((upsampled, bypass), 1)
    def forward(self, x):
        # Encode
        encode_block1 = self.conv_encode1(x)
        encode_pool1 = self.conv_maxpool1(encode_block1)
        encode_block2 = self.conv_encode2(encode_pool1)
        encode_pool2 = self.conv_maxpool2(encode_block2)
        encode_block3 = self.conv_encode3(encode_pool2)
        encode_pool3 = self.conv_maxpool3(encode_block3)
        encode_block4 = self.conv_encode4(encode_pool3)
        encode_pool4 = self.conv_maxpool4(encode_block4)
        # print('encode_block1 size', encode_block1.size())
        # print('encode_pool1 size', encode_pool1.size())
        # print('encode_block2 size', encode_block2.size())
        # print('encode_pool2 size', encode_pool2.size())
        # print('encode_block3 size', encode_block3.size())
        # print('encode_pool3 size', encode_pool3.size())
        # print('encode_block4 size', encode_block4.size())
        # print('encode_pool4 size', encode_pool4.size())
        # Bottleneck
        bottleneck1 = self.bottleneck(encode_pool4)
        # print('bottleneck1 size', bottleneck1.size())
        # print('encode_block4 size', encode_block4.size())
        decode_block4 = self.crop_and_concat(bottleneck1, encode_block4, crop=True)
        cat_layer4 = self.conv_decode4(decode_block4)
        decode_block3 = self.crop_and_concat(cat_layer4, encode_block3, crop=True)
        cat_layer3 = self.conv_decode3(decode_block3)
        decode_block2 = self.crop_and_concat(cat_layer3, encode_block2, crop=True)
        cat_layer2 = self.conv_decode2(decode_block2)
        decode_block1 = self.crop_and_concat(cat_layer2, encode_block1, crop=True)
        final_layer = self.final_layer(decode_block1)
        final_layer = torch.sigmoid(final_layer)
        return final_layer, decode_block1, decode_block2, decode_block3, decode_block4
# unet = Unet(in_channel=1, out_channel=2)
# inputs = Variable(torch.zeros(2, 1, 572, 572))
# outputs = unet(inputs)