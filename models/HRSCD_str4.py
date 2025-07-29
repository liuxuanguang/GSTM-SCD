import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

class ResBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        if not inplanes==planes or stride>1:
            self.downsample = nn.Sequential( conv1x1(inplanes, planes, stride), nn.BatchNorm2d(planes))
        else: self.downsample = None
    
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

class TransResBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=2):
        super(TransResBlock, self).__init__()
        self.conv1 = nn.ConvTranspose2d(inplanes, planes, kernel_size=2, stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        if stride>1:
            self.upsample = nn.Sequential(torch.nn.Upsample(scale_factor=2.0, mode='bilinear'),
                                          conv1x1(inplanes, planes), nn.BatchNorm2d(planes))
        else: self.upsample = None
    
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.upsample is not None:
            identity = self.upsample(x)

        out += identity
        out = self.relu(out)
        return out

class _EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super(_EncoderBlock, self).__init__()
        self.downsample = False
        if stride>1:
            self.downsample = True
            self.down_conv = ResBlock(in_channels, out_channels, stride)
            in_channels = out_channels
        self.encode = nn.Sequential(ResBlock(in_channels, out_channels),
                                    ResBlock(out_channels, out_channels))

    def forward(self, x):
        if self.downsample:
            x = self.down_conv(x)
        x = self.encode(x)
        return x

class _DecoderBlock(nn.Module):
    def __init__(self, in_ch1, in_ch2, mid_channels, out_channels):
        super(_DecoderBlock, self).__init__()
        self.up = TransResBlock(in_ch1, mid_channels)
        in_ch = mid_channels + in_ch2
        self.decode = nn.Sequential(ResBlock(in_ch, mid_channels),
                                    ResBlock(mid_channels, out_channels))

    def forward(self, x1, x2):
        x = self.up(x1)
        x = torch.cat((x, x2), dim=1)
        x = self.decode(x)        
        return x

class Encoder(nn.Module):
    def __init__(self, in_channels=3):
        super(Encoder, self).__init__()
        
        self.Enc0 = _EncoderBlock(in_channels, 8, stride=1)
        self.Enc1 = _EncoderBlock(8, 16)
        self.Enc2 = _EncoderBlock(16, 32)
        self.Enc3 = _EncoderBlock(32, 64)
        self.Enc4 = _EncoderBlock(64, 128)
        self.Enc5 = _EncoderBlock(128, 256)  

    def forward(self, x):
        x_size = x.size()
                
        enc0 = self.Enc0(x)
        enc1 = self.Enc1(enc0)
        enc2 = self.Enc2(enc1)
        enc3 = self.Enc3(enc2)
        enc4 = self.Enc4(enc3)
        enc5 = self.Enc5(enc4)
        
        encs = [enc0, enc1, enc2, enc3, enc4, enc5]
        
        return encs

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
                
        self.Dec4 = _DecoderBlock(256, 128, 128, 128)
        self.Dec3 = _DecoderBlock(128, 64, 64, 64)
        self.Dec2 = _DecoderBlock(64, 32, 32, 32)
        self.Dec1 = _DecoderBlock(32, 16, 16, 16)
        self.Dec0 = _DecoderBlock(16, 8, 8, 8)

    def forward(self, encs):
        enc0, enc1, enc2, enc3, enc4, enc5 = encs
        
        dec4 = self.Dec4(enc5, enc4)
        dec3 = self.Dec3(dec4, enc3)
        dec2 = self.Dec2(dec3, enc2)
        dec1 = self.Dec1(dec2, enc1)
        dec0 = self.Dec0(dec1, enc0)        
        return dec0

class Decoder_cd(nn.Module):
    def __init__(self):
        super(Decoder_cd, self).__init__()
                
        self.Dec4 = _DecoderBlock(256, 128*2, 128, 128)
        self.Dec3 = _DecoderBlock(128, 64*2, 64, 64)
        self.Dec2 = _DecoderBlock(64, 32*2, 32, 32)
        self.Dec1 = _DecoderBlock(32, 16*2, 16, 16)
        self.Dec0 = _DecoderBlock(16, 8*2, 8, 8)

    def forward(self, encs_cd, encs1, encs2):
        enc0_cd, enc1_cd, enc2_cd, enc3_cd, enc4_cd, enc5_cd = encs_cd
        f1_enc0, f1_enc1, f1_enc2, f1_enc3, f1_enc4, _ = encs1
        f2_enc0, f2_enc1, f2_enc2, f2_enc3, f2_enc4, _ = encs2
        
        dec4 = self.Dec4(enc5_cd, torch.cat([enc4_cd, torch.abs(f1_enc4-f2_enc4)], 1))
        dec3 = self.Dec3(dec4, torch.cat([enc3_cd, torch.abs(f1_enc3-f2_enc3)], 1))
        dec2 = self.Dec2(dec3, torch.cat([enc2_cd, torch.abs(f1_enc2-f2_enc2)], 1))
        dec1 = self.Dec1(dec2, torch.cat([enc1_cd, torch.abs(f1_enc1-f2_enc1)], 1))
        dec0 = self.Dec0(dec1, torch.cat([enc0_cd, torch.abs(f1_enc0-f2_enc0)], 1))    
        return dec0

class HRSCD_str4(nn.Module):
    def __init__(self, in_channels=3, num_classes=1):
        super(HRSCD_str4, self).__init__()
        self.encoder = Encoder(in_channels)
        self.decoder = Decoder()
        self.encoder_cd = Encoder(in_channels*2)
        self.decoder_cd = Decoder_cd()
        
        self.classifier = nn.Sequential( ResBlock(8, 8), nn.Conv2d(8, num_classes, kernel_size=1) )
        self.classifier_CD = nn.Sequential( ResBlock(8, 8), nn.Conv2d(8, 1, kernel_size=1) )
        
    def forward(self, x1, x2):
        encs1 = self.encoder(x1)
        encs2 = self.encoder(x2)
        
        x = torch.cat([x1, x2], 1)
        encs_cd = self.encoder_cd(x)
        
        out1 = self.decoder(encs1)
        out2 = self.decoder(encs2)
        out_cd = self.decoder_cd(encs_cd, encs1, encs2)
        
        out1 = self.classifier(out1)
        out2 = self.classifier(out2)
        out_cd = self.classifier_CD(out_cd)
        
        return out1, out2, out_cd.squeeze(1)


class HRSCD_str4_MT(nn.Module):
    def __init__(self, in_channels=4, num_classes=12):
        super(HRSCD_str4_MT, self).__init__()
        self.encoder = Encoder(in_channels)
        self.decoder = Decoder()
        self.encoder_cd = Encoder(in_channels * 2)
        self.decoder_cd = Decoder_cd()

        self.softmax = nn.Softmax(dim=1)

        self.classifier = nn.Sequential(ResBlock(8, 8), nn.Conv2d(8, num_classes, kernel_size=1))
        self.classifier_CD = nn.Sequential(ResBlock(8, 8), nn.Conv2d(8, 1, kernel_size=1))

    def forward(self, x1, x2, x3):
        x1 = x1.float()
        x2 = x2.float()
        x3 = x3.float()
        encs1 = self.encoder(x1)
        encs2 = self.encoder(x2)
        encs3 = self.encoder(x3)
        x = torch.cat([x1, x3], 1)
        encs_cd = self.encoder_cd(x)

        out1 = self.decoder(encs1)
        out2 = self.decoder(encs2)
        out3 = self.decoder(encs3)
        out_cd = self.decoder_cd(encs_cd, encs1, encs2)

        out1 = self.classifier(out1)
        out2 = self.classifier(out2)
        out3 = self.classifier(out3)
        out_cd = self.classifier_CD(out_cd)

        out1 = self.softmax(out1)
        out2 = self.softmax(out2)
        out3 = self.softmax(out3)
        out_cd = torch.sigmoid(out_cd)

        return out1, out2, out3, out_cd.squeeze(1)


# class HRSCD_str4_NMT(nn.Module):
#     def __init__(self, in_channels=4, num_classes=7):
#         super(HRSCD_str4_NMT, self).__init__()
#         self.encoder = Encoder(in_channels)
#         self.decoder = Decoder()
#         self.encoder_cd = Encoder(in_channels * 2)
#         self.decoder_cd = Decoder_cd()
#
#         self.softmax = nn.Softmax(dim=1)
#
#         self.classifier = nn.Sequential(ResBlock(8, 8), nn.Conv2d(8, num_classes, kernel_size=1))
#         self.classifier_CD = nn.Sequential(ResBlock(8, 8), nn.Conv2d(8, 1, kernel_size=1))
#
#     def forward(self, x1, x2, x3, x4, x5, x6):
#         x1 = x1.float()
#         x2 = x2.float()
#         x3 = x3.float()
#         x4 = x4.float()
#         x5 = x5.float()
#         x6 = x6.float()
#
#         encs1 = self.encoder(x1)
#         encs2 = self.encoder(x2)
#         encs3 = self.encoder(x3)
#         encs4 = self.encoder(x4)
#         encs5 = self.encoder(x5)
#         encs6 = self.encoder(x6)
#         x = torch.cat([x1, x6], 1)
#         encs_cd = self.encoder_cd(x)
#
#         out1 = self.decoder(encs1)
#         out2 = self.decoder(encs2)
#         out3 = self.decoder(encs3)
#         out4 = self.decoder(encs4)
#         out5 = self.decoder(encs5)
#         out6 = self.decoder(encs6)
#         out_cd = self.decoder_cd(encs_cd, encs1, encs6)
#
#         out1 = self.classifier(out1)
#         out2 = self.classifier(out2)
#         out3 = self.classifier(out3)
#         out4 = self.classifier(out4)
#         out5 = self.classifier(out5)
#         out6 = self.classifier(out6)
#         out_cd = self.classifier_CD(out_cd)
#
#         out1 = self.softmax(out1)
#         out2 = self.softmax(out2)
#         out3 = self.softmax(out3)
#         out4 = self.softmax(out4)
#         out5 = self.softmax(out5)
#         out6 = self.softmax(out6)
#         out_cd = torch.sigmoid(out_cd)
#         pair = [0, 5]
#
#         return [out1, out2, out3, out4, out5, out6], out_cd.squeeze(1), pair
class HRSCD_str4_NMT(nn.Module):
    def __init__(self, in_channels=4, num_classes=8):
        super(HRSCD_str4_NMT, self).__init__()
        self.encoder = Encoder(in_channels)
        self.decoder = Decoder()
        self.encoder_cd = Encoder(in_channels * 2)
        self.decoder_cd = Decoder_cd()

        self.softmax = nn.Softmax(dim=1)

        self.classifier = nn.Sequential(ResBlock(8, 8), nn.Conv2d(8, num_classes, kernel_size=1))
        self.classifier_CD = nn.Sequential(ResBlock(8, 8), nn.Conv2d(8, 1, kernel_size=1))

    def forward(self, x1, x2, x3, x4, x5, x6, pair):
        x1 = x1.float()
        x2 = x2.float()
        x3 = x3.float()
        x4 = x4.float()
        x5 = x5.float()
        x6 = x6.float()
        xs = [x1, x2, x3, x4, x5, x6]
        encs1 = self.encoder(x1)
        encs2 = self.encoder(x2)
        encs3 = self.encoder(x3)
        encs4 = self.encoder(x4)
        encs5 = self.encoder(x5)
        encs6 = self.encoder(x6)
        encs_s = [encs1, encs2, encs3, encs4, encs5, encs6]
        x = torch.cat([xs[pair[0]], xs[pair[-1]]], 1)
        encs_cd = self.encoder_cd(x)

        out1 = self.decoder(encs1)
        out2 = self.decoder(encs2)
        out3 = self.decoder(encs3)
        out4 = self.decoder(encs4)
        out5 = self.decoder(encs5)
        out6 = self.decoder(encs6)
        out_cd = self.decoder_cd(encs_cd, encs_s[pair[0]], encs_s[pair[-1]])

        out1 = self.classifier(out1)
        out2 = self.classifier(out2)
        out3 = self.classifier(out3)
        out4 = self.classifier(out4)
        out5 = self.classifier(out5)
        out6 = self.classifier(out6)
        out_cd = self.classifier_CD(out_cd)

        out1 = self.softmax(out1)
        out2 = self.softmax(out2)
        out3 = self.softmax(out3)
        out4 = self.softmax(out4)
        out5 = self.softmax(out5)
        out6 = self.softmax(out6)
        out_cd = torch.sigmoid(out_cd)

        # return [out1, out2, out3, out4, out5, out6], out_cd.squeeze(1)
        return out1, out2, out3, out4, out5, out6, out_cd.squeeze(1)

class HRSCD_str4_NMT_infer(nn.Module):
    def __init__(self, in_channels=4, num_classes=7):
        super(HRSCD_str4_NMT_infer, self).__init__()
        self.encoder = Encoder(in_channels)
        self.decoder = Decoder()
        self.encoder_cd = Encoder(in_channels * 2)
        self.decoder_cd = Decoder_cd()

        self.softmax = nn.Softmax(dim=1)

        self.classifier = nn.Sequential(ResBlock(8, 8), nn.Conv2d(8, num_classes, kernel_size=1))
        self.classifier_CD = nn.Sequential(ResBlock(8, 8), nn.Conv2d(8, 1, kernel_size=1))

    def forward(self, x1, x2, x3, x4, x5, x6):
        x1 = x1.float()
        x2 = x2.float()
        x3 = x3.float()
        x4 = x4.float()
        x5 = x5.float()
        x6 = x6.float()

        encs1 = self.encoder(x1)
        encs2 = self.encoder(x2)
        encs3 = self.encoder(x3)
        encs4 = self.encoder(x4)
        encs5 = self.encoder(x5)
        encs6 = self.encoder(x6)
        x = torch.cat([x1, x6], 1)
        encs_cd = self.encoder_cd(x)

        out1 = self.decoder(encs1)
        out2 = self.decoder(encs2)
        out3 = self.decoder(encs3)
        out4 = self.decoder(encs4)
        out5 = self.decoder(encs5)
        out6 = self.decoder(encs6)
        out_cd = self.decoder_cd(encs_cd, encs1, encs6)

        out1 = self.classifier(out1)
        out2 = self.classifier(out2)
        out3 = self.classifier(out3)
        out4 = self.classifier(out4)
        out5 = self.classifier(out5)
        out6 = self.classifier(out6)
        out_cd = self.classifier_CD(out_cd)

        out1 = self.softmax(out1)
        out2 = self.softmax(out2)
        out3 = self.softmax(out3)
        out4 = self.softmax(out4)
        out5 = self.softmax(out5)
        out6 = self.softmax(out6)
        out_cd = torch.sigmoid(out_cd)
        pair = [0, 5]

        return [out1, out2, out3, out4, out5, out6], out_cd.squeeze(1), pair