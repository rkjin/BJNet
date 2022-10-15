from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
from models.submodule import *
import math


class feature_extraction(nn.Module):
    def __init__(self, concat_feature=False, concat_feature_channel=12): #True, 12
        super(feature_extraction, self).__init__()
        self.concat_feature = concat_feature

        self.inplanes = 32
        self.firstconv = nn.Sequential(convbn(3, 32, 3, 2, 1, 1),
                                       Mish(),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       Mish(),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       Mish())

        # self.layer1 = self._make_layer(BasicBlock, 32, 3, 1, 1, 1)
        self.layer2 = self._make_layer(BasicBlock, 64, 1, 1, 1, 1)
        self.layer3 = self._make_layer(BasicBlock, 128, 1, 2, 1, 1)
        self.layer4 = self._make_layer(BasicBlock, 192, 1, 2, 1, 1)
        self.layer5 = self._make_layer(BasicBlock, 256, 1, 2, 1, 1)
        self.layer6 = self._make_layer(BasicBlock, 512, 1, 2, 1, 1)
        self.pyramid_pooling = pyramidPooling(512, None, fusion_mode='sum', model_name='icnet')
        self.upconv6 = nn.Sequential(nn.Upsample(scale_factor=2),
                                     convbn(512, 256, 3, 1, 1, 1),
                                     Mish())
        self.iconv5 = nn.Sequential(convbn(512, 256, 3, 1, 1, 1),
                                    Mish())
        self.upconv5 = nn.Sequential(nn.Upsample(scale_factor=2),
                                     convbn(256, 192, 3, 1, 1, 1),
                                     Mish())
        self.iconv4 = nn.Sequential(convbn(384, 192, 3, 1, 1, 1),
                                    Mish())
        self.upconv4 = nn.Sequential(nn.Upsample(scale_factor=2),
                                     convbn(192, 128, 3, 1, 1, 1),
                                     Mish())
        self.iconv3 = nn.Sequential(convbn(256, 128, 3, 1, 1, 1),
                                    Mish())
        self.upconv3 = nn.Sequential(nn.Upsample(scale_factor=2),
                                     convbn(128, 64, 3, 1, 1, 1),
                                     Mish())
        self.iconv2 = nn.Sequential(convbn(128, 64, 3, 1, 1, 1),
                                    Mish())
        # self.upconv2 = nn.Sequential(nn.Upsample(scale_factor=2),
        #                              convbn(64, 32, 3, 1, 1, 1),
        #                              nn.ReLU(inplace=True))

        # self.gw1 = nn.Sequential(convbn(32, 40, 3, 1, 1, 1),
        #                          nn.ReLU(inplace=True),
        #                          nn.Conv2d(40, 40, kernel_size=1, padding=0, stride=1,
        #                                    bias=False))

        self.gw2 = nn.Sequential(convbn(64, 80, 3, 1, 1, 1),
                                          Mish(),
                                          nn.Conv2d(80, 80, kernel_size=1, padding=0, stride=1,
                                                    bias=False))

        self.gw3 = nn.Sequential(convbn(128, 160, 3, 1, 1, 1),
                                 Mish(),
                                 nn.Conv2d(160, 160, kernel_size=1, padding=0, stride=1,
                                           bias=False))

        self.gw4 = nn.Sequential(convbn(192, 160, 3, 1, 1, 1),
                                 Mish(),
                                 nn.Conv2d(160, 160, kernel_size=1, padding=0, stride=1,
                                           bias=False))

        self.gw5 = nn.Sequential(convbn(256, 320, 3, 1, 1, 1),
                                 Mish(),
                                 nn.Conv2d(320, 320, kernel_size=1, padding=0, stride=1,
                                           bias=False))

        self.gw6 = nn.Sequential(convbn(512, 320, 3, 1, 1, 1),
                                 Mish(),
                                 nn.Conv2d(320, 320, kernel_size=1, padding=0, stride=1,
                                           bias=False))
#        print('feature_extraction', self.concat_feature)
        if self.concat_feature:
            # self.concat1 = nn.Sequential(convbn(32, 16, 3, 1, 1, 1),
            #                              nn.ReLU(inplace=True),
            #                              nn.Conv2d(16, concat_feature_channel // 4, kernel_size=1, padding=0, stride=1,
            #                                        bias=False))

            self.concat2 = nn.Sequential(convbn(64, 32, 3, 1, 1, 1),
                                          Mish(),
                                          nn.Conv2d(32, concat_feature_channel // 2, kernel_size=1, padding=0, stride=1,
                                                    bias=False))
            self.concat3 = nn.Sequential(convbn(128, 128, 3, 1, 1, 1),
                                          Mish(),
                                          nn.Conv2d(128, concat_feature_channel, kernel_size=1, padding=0, stride=1,
                                                    bias=False))

            self.concat4 = nn.Sequential(convbn(192, 128, 3, 1, 1, 1),
                                          Mish(),
                                          nn.Conv2d(128, concat_feature_channel, kernel_size=1, padding=0, stride=1,
                                                    bias=False))

            self.concat5 = nn.Sequential(convbn(256, 128, 3, 1, 1, 1),
                                         Mish(),
                                         nn.Conv2d(128, concat_feature_channel, kernel_size=1, padding=0, stride=1,
                                                   bias=False))

            self.concat6 = nn.Sequential(convbn(512, 128, 3, 1, 1, 1),
                                         Mish(),
                                         nn.Conv2d(128, concat_feature_channel, kernel_size=1, padding=0, stride=1,
                                                   bias=False))




                   # (BasicBlock, 64,      1,      1,    1,      1)
    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion), )

        layers = [] # BssicBlock  32        64~512   1~2  None or Seq   1      1
        layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, pad, dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.firstconv(x)                #     [1, 3,  256, 512] 
        # x = self.layer1(x)                #      [1, 32, 128, 256]
        l2 = self.layer2(x)     #1/2 l2 torch.Size([1, 64, 128, 256])
        l3 = self.layer3(l2)    #1/4    torch.Size([1, 128, 64, 128])
        l4 = self.layer4(l3)    #1/8    torch.Size([1, 192, 32, 64])
        l5 = self.layer5(l4)    #1/16   torch.Size([1, 256, 16, 32])
        l6 = self.layer6(l5)    #1/32   torch.Size([1, 512, 8, 16])
        l6 = self.pyramid_pooling(l6)             #[1, 256, 8, 16]
         
        concat5 = torch.cat((l5, self.upconv6(l6)), dim=1)        #[1, 512, 16, 32]
        decov_5 = self.iconv5(concat5)                             #[1, 256, 16, 32]
        concat4 = torch.cat((l4, self.upconv5(decov_5)), dim=1)   #[1, 384, 32, 64]
        # concat4 = torch.cat((l4, self.upconv5(l5)), dim=1)
        decov_4 = self.iconv4(concat4)                           #  [1, 192, 32, 64]
        concat3 = torch.cat((l3, self.upconv4(decov_4)), dim=1)  # [1, 256, 64, 128]
        decov_3 = self.iconv3(concat3)                           #[1, 128, 64, 128]
        concat2 = torch.cat((l2, self.upconv3(decov_3)), dim=1)  # [1, 128, 128, 256]
        decov_2 = self.iconv2(concat2)                           # [1, 64, 128, 256]
        # decov_1 = self.upconv2(decov_2)


        # gw1 = self.gw1(decov_1)
        gw2 = self.gw2(decov_2) #[1, 80, 128, 256]
        gw3 = self.gw3(decov_3) #[1, 160, 64, 128]
        gw4 = self.gw4(decov_4) #[1, 160, 32, 64]
        gw5 = self.gw5(decov_5) #[1, 320, 16, 32]
        gw6 = self.gw6(l6)      #[1, 320, 8, 16]

        if not self.concat_feature:
            return {"gw2": gw2, "gw3": gw3, "gw4": gw4}
        else:
            # concat_feature1 = self.concat1(decov_1)
            concat_feature2 = self.concat2(decov_2) #[1, 6, 128, 256]
            concat_feature3 = self.concat3(decov_3)#[1, 12, 64, 128]
            concat_feature4 = self.concat4(decov_4)#[1, 12, 32, 64]
            concat_feature5 = self.concat5(decov_5)#[1, 12, 16, 32]
            concat_feature6 = self.concat6(l6)     #[1, 12, 8, 16]
            return {"gw2": gw2, "gw3": gw3, "gw4": gw4, "gw5": gw5, "gw6": gw6,
                    "concat_feature2": concat_feature2, "concat_feature3": concat_feature3, "concat_feature4": concat_feature4,
                    "concat_feature5": concat_feature5, "concat_feature6": concat_feature6}

class hourglassup(nn.Module):
    def __init__(self, in_channels): # in_channels : 32
        super(hourglassup, self).__init__()

        self.conv1 = nn.Conv3d(in_channels, in_channels * 2, kernel_size=3, stride=2,
                                   padding=1, bias=False)

        self.conv2 = nn.Sequential(convbn_3d(in_channels * 2, in_channels * 2, 3, 1, 1),
                                   Mish())

        self.conv3 = nn.Conv3d(in_channels * 2, in_channels * 4, kernel_size=3, stride=2,
                               padding=1, bias=False)

        self.conv4 = nn.Sequential(convbn_3d(in_channels * 4, in_channels * 4, 3, 1, 1),
                                   Mish())

        self.conv8 = nn.Sequential(
            nn.ConvTranspose3d(in_channels * 4, in_channels * 2, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(in_channels * 2))

        self.conv9 = nn.Sequential(
            nn.ConvTranspose3d(in_channels * 2, in_channels, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(in_channels))

        self.combine1 = nn.Sequential(convbn_3d(in_channels * 4, in_channels * 2, 3, 1, 1),
                                   Mish())
        self.combine2 = nn.Sequential(convbn_3d(in_channels * 6, in_channels * 4, 3, 1, 1),
                                      Mish())
        self.combine3 = nn.Sequential(convbn_3d(in_channels * 6, in_channels * 4, 3, 1, 1),
                                      Mish())

        self.redir1 = convbn_3d(in_channels, in_channels, kernel_size=1, stride=1, pad=0)
#        self.redir2 = convbn_3d(in_channels * 2, in_channels * 2, kernel_size=1, stride=1, pad=0)
        self.redir2 = convbn_3d(in_channels * 2, in_channels * 2, kernel_size=1, stride=1, pad=0)        
        self.redir3 = convbn_3d(in_channels * 4, in_channels * 4, kernel_size=1, stride=1, pad=0)


    def forward(self, x, feature4, feature5): #[1, 32, 32, 32, 64]) torch.Size([1, 64, 16, 16, 32]) torch.Size([1, 64, 8, 8, 16])
        conv1 = self.conv1(x)          #1/8 [1, 64, 16, 16, 32])
        conv1 = torch.cat((conv1, feature4), dim=1)   #1/8  [1, 128, 16, 16, 32])
        conv1 = self.combine1(conv1)   #1/8 [1, 64, 16, 16, 32])
        conv2 = self.conv2(conv1)      #1/8 [1, 64, 16, 16, 32])

        conv3 = self.conv3(conv2)      #1/16 [1, 128, 8, 8, 16])
        conv3 = torch.cat((conv3, feature5), dim=1)   #1/16 [1, 192, 8, 8, 16])
        conv3 = self.combine2(conv3)   #1/16 [1, 128, 8, 8, 16])
        conv4 = self.conv4(conv3)      #1/16 [1, 128, 8, 8, 16])

        conv8 = FMish(self.conv8(conv4) + self.redir2(conv2)) #[1, 64, 16, 16, 32])
        conv9 = FMish(self.conv9(conv8) + self.redir1(x))#[1, 32, 32, 32, 64])


        return conv9

class hourglass(nn.Module):
    def __init__(self, in_channels):
        super(hourglass, self).__init__()

        self.conv1 = nn.Sequential(convbn_3d(in_channels, in_channels * 2, 3, 2, 1),
                                   Mish())

        self.conv2 = nn.Sequential(convbn_3d(in_channels * 2, in_channels * 2, 3, 1, 1),
                                   Mish())

        self.conv3 = nn.Sequential(convbn_3d(in_channels * 2, in_channels * 4, 3, 2, 1),
                                   Mish())

        self.conv4 = nn.Sequential(convbn_3d(in_channels * 4, in_channels * 4, 3, 1, 1),
                                   Mish())

        self.conv5 = nn.Sequential(
            nn.ConvTranspose3d(in_channels * 4, in_channels * 2, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(in_channels * 2))

        self.conv6 = nn.Sequential(
            nn.ConvTranspose3d(in_channels * 2, in_channels, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(in_channels))

        self.redir1 = convbn_3d(in_channels, in_channels, kernel_size=1, stride=1, pad=0)
        self.redir2 = convbn_3d(in_channels * 2, in_channels * 2, kernel_size=1, stride=1, pad=0)

    def forward(self, x): #[1, 32, 32, 32, 64])
        conv1 = self.conv1(x) #[1, 64, 16, 16, 32])
        conv2 = self.conv2(conv1)#[1, 64, 16, 16, 32])

        conv3 = self.conv3(conv2)#[1, 128, 8, 8, 16])
        conv4 = self.conv4(conv3)#[1, 128, 8, 8, 16])

        # conv5 = F.relu(self.conv5(conv4) + self.redir2(conv2), inplace=True)
        # conv6 = F.relu(self.conv6(conv5) + self.redir1(x), inplace=True)

        conv5 = FMish(self.conv5(conv4) + self.redir2(conv2))#[1, 64, 16, 16, 32])
        conv6 = FMish(self.conv6(conv5) + self.redir1(x))#[1, 32, 32, 32, 64])

        return conv6

class cfnet(nn.Module):
    def __init__(self, maxdisp, use_concat_volume=False):
        super(cfnet, self).__init__()
        self.maxdisp = maxdisp
        self.use_concat_volume = use_concat_volume
        self.v_scale_s1 = 1
        self.v_scale_s2 = 2
        self.v_scale_s3 = 3
        self.sample_count_s1 = 6
        self.sample_count_s2 = 10
        self.sample_count_s3 = 14
        self.num_groups = 40
        self.uniform_sampler = UniformSampler()
        self.spatial_transformer = SpatialTransformer()

        #print(concat_volume, concat_channels)
        if self.use_concat_volume: #True
            self.concat_channels = 12
            self.feature_extraction = feature_extraction(concat_feature=True,
                                                         concat_feature_channel=self.concat_channels)
        else:
            self.concat_channels = 0
            self.feature_extraction = feature_extraction(concat_feature=False)

        self.dres0 = nn.Sequential(convbn_3d(self.num_groups + self.concat_channels*2, 32, 3, 1, 1),
                                   Mish(),
                                   convbn_3d(32, 32, 3, 1, 1),
                                   Mish())

        self.dres1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   Mish(),
                                   convbn_3d(32, 32, 3, 1, 1))


        self.dres0_5 = nn.Sequential(convbn_3d(self.num_groups + self.concat_channels*2, 64, 3, 1, 1),
                                   Mish(),
                                   convbn_3d(64, 64, 3, 1, 1),
                                   Mish())

        self.dres1_5 = nn.Sequential(convbn_3d(64, 64, 3, 1, 1),
                                   Mish(),
                                   convbn_3d(64, 64, 3, 1, 1))

        self.dres0_6 = nn.Sequential(convbn_3d(self.num_groups + self.concat_channels*2, 64, 3, 1, 1),
                                   Mish(),
                                   convbn_3d(64, 64, 3, 1, 1),
                                   Mish())

        self.dres1_6 = nn.Sequential(convbn_3d(64, 64, 3, 1, 1),
                                   Mish(),
                                   convbn_3d(64, 64, 3, 1, 1))

        self.combine1 = hourglassup(32)

        # self.dres2 = hourglass(32)

        self.dres3 = hourglass(32)

        # self.dres4 = hourglass(32)

        self.confidence0_s3 = nn.Sequential(convbn_3d(self.num_groups + self.concat_channels*2 + 1 , 32, 3, 1, 1),
                                   Mish(),
                                   convbn_3d(32, 32, 3, 1, 1),
                                   Mish())

        self.confidence1_s3 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   Mish(),
                                   convbn_3d(32, 32, 3, 1, 1))

        self.confidence2_s3 = hourglass(32)

        self.confidence3_s3 = hourglass(32)

        self.confidence0_s2 = nn.Sequential(convbn_3d(self.num_groups//2 + self.concat_channels + 1, 16, 3, 1, 1),
                                   Mish(),
                                   convbn_3d(16, 16, 3, 1, 1),
                                   Mish())

        self.confidence1_s2 = nn.Sequential(convbn_3d(16, 16, 3, 1, 1),
                                   Mish(),
                                   convbn_3d(16, 16, 3, 1, 1))

        self.confidence2_s2 = hourglass(16)

        self.confidence3_s2 = hourglass(16)


        # self.confidence0_s1 = nn.Sequential(convbn_3d(self.num_groups // 4 + self.concat_channels // 2 + 1, 16, 3, 1, 1),
        #                                     nn.ReLU(inplace=True),
        #                                     convbn_3d(16, 16, 3, 1, 1),
        #                                     nn.ReLU(inplace=True))
        #
        # self.confidence1_s1 = nn.Sequential(convbn_3d(16, 16, 3, 1, 1),
        #                                     nn.ReLU(inplace=True),
        #                                     convbn_3d(16, 16, 3, 1, 1))
        #
        # self.confidence2_s1 = hourglass(16)

        # self.confidence3 = hourglass(32)
        #
        # self.confidence4 = hourglass(32)

        self.confidence_classif0_s3 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                                    Mish(),
                                                    nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.confidence_classif1_s3 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      Mish(),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.confidence_classifmid_s3 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                                    Mish(),
                                                    nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.confidence_classif0_s2 = nn.Sequential(convbn_3d(16, 16, 3, 1, 1),
                                                    Mish(),
                                                    nn.Conv3d(16, 1, kernel_size=3, padding=1, stride=1, bias=False))


        self.confidence_classif1_s2 = nn.Sequential(convbn_3d(16, 16, 3, 1, 1),
                                                    Mish(),
                                                    nn.Conv3d(16, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.confidence_classifmid_s2 = nn.Sequential(convbn_3d(16, 16, 3, 1, 1),
                                                    Mish(),
                                                    nn.Conv3d(16, 1, kernel_size=3, padding=1, stride=1, bias=False))

        # self.confidence_classif1_s1 = nn.Sequential(convbn_3d(16, 16, 3, 1, 1),
        #                                             nn.ReLU(inplace=True),
        #                                             nn.Conv3d(16, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif0 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      Mish(),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      Mish(),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif2 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      Mish(),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.gamma_s3 = nn.Parameter(torch.zeros(1))
        self.beta_s3 = nn.Parameter(torch.zeros(1))
        self.gamma_s2 = nn.Parameter(torch.zeros(1))
        self.beta_s2 = nn.Parameter(torch.zeros(1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            # elif isinstance(m, nn.Linear):
                # m.bias.data.zero_()

    def generate_search_range(self, sample_count, input_min_disparity, input_max_disparity, scale):
        """
        Description:    Generates the disparity search range.

        Returns:
            :min_disparity: Lower bound of disparity search range
            :max_disparity: Upper bound of disaprity search range.
        """

        min_disparity = torch.clamp(input_min_disparity - torch.clamp((
                sample_count - input_max_disparity + input_min_disparity), min=0) / 2.0, min=0, max=self.maxdisp // (2**scale) - 1)
        max_disparity = torch.clamp(input_max_disparity + torch.clamp(
                sample_count - input_max_disparity + input_min_disparity, min=0) / 2.0, min=0, max=self.maxdisp // (2**scale) - 1)

        return min_disparity, max_disparity

    def generate_disparity_samples(self, min_disparity, max_disparity, sample_count=12):
        """
        Description:    Generates "sample_count" number of disparity samples from the
                                                            search range (min_disparity, max_disparity)
                        Samples are generated by uniform sampler

        Args:
            :min_disparity: LowerBound of the disaprity search range.
            :max_disparity: UpperBound of the disparity search range.
            :sample_count: Number of samples to be generated from the input search range.

        Returns:
            :disparity_samples:
        """
        disparity_samples = self.uniform_sampler(min_disparity, max_disparity, sample_count)

        disparity_samples = torch.cat((torch.floor(min_disparity), disparity_samples, torch.ceil(max_disparity)),
                                      dim=1).long()                   # disparity level = sample_count + 2
        return disparity_samples

    def cost_volume_generator(self, left_input, right_input, disparity_samples, model = 'concat', num_groups = 40):
        """
        Description: Generates cost-volume using left image features, disaprity samples
                                                            and warped right image features.
        Args:
            :left_input: Left Image fetaures
            :right_input: Right Image features
            :disparity_samples: Disaprity samples
            :model : concat or group correlation

        Returns:
            :cost_volume:
            :disaprity_samples:
        """

        right_feature_map, left_feature_map = self.spatial_transformer(left_input,
                                                                       right_input, disparity_samples)
        disparity_samples = disparity_samples.unsqueeze(1).float()
        if model == 'concat':
             cost_volume = torch.cat((left_feature_map, right_feature_map), dim=1)
        else:
             cost_volume = groupwise_correlation_4D(left_feature_map, right_feature_map, num_groups)

        return cost_volume, disparity_samples

    def forward(self, left, right):
        features_left = self.feature_extraction(left)
        features_right = self.feature_extraction(right)
#       group wise correlation 
        gwc_volume4 = build_gwc_volume(features_left["gw4"], features_right["gw4"], self.maxdisp // 8,
                                       self.num_groups)   #[1, 40, 32, 32, 64])

        gwc_volume5 = build_gwc_volume(features_left["gw5"], features_right["gw5"], self.maxdisp // 16,
                                       self.num_groups)   #[1, 40, 16, 16, 32])

        gwc_volume6 = build_gwc_volume(features_left["gw6"], features_right["gw6"], self.maxdisp // 32,
                                       self.num_groups)   # [1, 40, 8, 8, 16])

#        print('cfnet use concat vol', self.use_concat_volume)                                       
        if self.use_concat_volume: 
                        
            concat_volume4 = build_concat_volume(features_left["concat_feature4"], features_right["concat_feature4"],
                                                 self.maxdisp // 8) #[1, 24, 32, 32, 64])
            concat_volume5 = build_concat_volume(features_left["concat_feature5"], features_right["concat_feature5"],
                                                 self.maxdisp // 16) #[1, 24, 16, 16, 32])
            concat_volume6 = build_concat_volume(features_left["concat_feature6"], features_right["concat_feature6"],
                                                 self.maxdisp // 32) #[1, 24, 8, 8, 16])

            volume4 = torch.cat((gwc_volume4, concat_volume4), 1) #[1, 64, 32, 32, 64])
            volume5 = torch.cat((gwc_volume5, concat_volume5), 1)#[1, 64, 16, 16, 32])
            volume6 = torch.cat((gwc_volume6, concat_volume6), 1)#[1, 64, 8, 8, 16])

        else:
            volume4 = gwc_volume4

        cost0_4 = self.dres0(volume4)              #[1, 32, 32, 32, 64])
        cost0_4 = self.dres1(cost0_4) + cost0_4    #[1, 32, 32, 32, 64])
        cost0_5 = self.dres0_5(volume5)              # [1, 64, 16, 16, 32])
        cost0_5 = self.dres1_5(cost0_5) + cost0_5 # [1, 64, 16, 16, 32])
        cost0_6 = self.dres0_6(volume6)            #[1, 64, 8, 8, 16])
        cost0_6 = self.dres1_6(cost0_6) + cost0_6    #[1, 64, 8, 8, 16])
        out1_4 = self.combine1(cost0_4, cost0_5, cost0_6)#[1, 32, 32, 32, 64]) hourglassup
        out2_4 = self.dres3(out1_4)                     #[1, 32, 32, 32, 64]) hourglass

        ########   
        cost2_s4 = self.classif2(out2_4)#[1, 1, 32, 32, 64])
        cost2_s4 = torch.squeeze(cost2_s4, 1)#[1, 32, 32, 64])
        pred2_possibility_s4 = F.softmax(cost2_s4, dim=1)#[1, 32, 32, 64])
        pred2_s4 = disparity_regression(pred2_possibility_s4, self.maxdisp // 8).unsqueeze(1)#[1, 1, 32, 64])
        
        pred2_s4_cur = pred2_s4.detach()#[1, 1, 32, 64]) False
        pred2_v_s4 = disparity_variance(pred2_possibility_s4, self.maxdisp // 8, pred2_s4_cur)  # get the variance[1, 1, 32, 64]) True
        pred2_v_s4 = pred2_v_s4.sqrt()#[1, 1, 32, 64]) True
        mindisparity_s3 = pred2_s4_cur - (self.gamma_s3 + 1) * pred2_v_s4 - self.beta_s3 #([1, 1, 32, 64]) True
        maxdisparity_s3 = pred2_s4_cur + (self.gamma_s3 + 1) * pred2_v_s4 + self.beta_s3 #([1, 1, 32, 64]) True
        maxdisparity_s3 = F.upsample(maxdisparity_s3 * 2, [left.size()[2] // 4, left.size()[3] // 4], mode='bilinear',
                                    align_corners=True) #([1, 1, 64, 128]) ?????????????????  True
        mindisparity_s3 = F.upsample(mindisparity_s3 * 2, [left.size()[2] // 4, left.size()[3] // 4], mode='bilinear',
                                    align_corners=True) #([1, 1, 64, 128]) ??????????????????  True
                                                                                          # 14 + 1   
        mindisparity_s3_1, maxdisparity_s3_1 = self.generate_search_range(self.sample_count_s3 + 1, mindisparity_s3, maxdisparity_s3, scale = 2)# True
        #[1, 1, 64, 128]) torch.Size([1, 1, 64, 128])
        disparity_samples_s3 = self.generate_disparity_samples(mindisparity_s3_1, maxdisparity_s3_1, self.sample_count_s3).float()#[1, 16, 64, 128])False
        confidence_v_concat_s3, _ = self.cost_volume_generator(features_left["concat_feature3"],
                                                            features_right["concat_feature3"], disparity_samples_s3, 'concat')#[1, 24, 16, 64, 128]) True
        confidence_v_gwc_s3, disparity_samples_s3 = self.cost_volume_generator(features_left["gw3"], features_right["gw3"],
                                                                         disparity_samples_s3, 'gwc', self.num_groups)#[1, 40, 16, 64, 128]) torch.Size([1, 1, 16, 64, 128]) True
        confidence_v_s3 = torch.cat((confidence_v_gwc_s3, confidence_v_concat_s3, disparity_samples_s3), dim=1)#[1, 65, 16, 64, 128]) True

        disparity_samples_s3 = torch.squeeze(disparity_samples_s3, dim=1)#[1, 16, 64, 128]) False

        cost0_s3 = self.confidence0_s3(confidence_v_s3)#[1, 32, 16, 64, 128])
        cost0_s3 = self.confidence1_s3(cost0_s3) + cost0_s3#[1, 32, 16, 64, 128])

        out1_s3 = self.confidence2_s3(cost0_s3)#[1, 32, 16, 64, 128])
        out2_s3 = self.confidence3_s3(out1_s3)#[1, 32, 16, 64, 128])

        cost1_s3 = self.confidence_classif1_s3(out2_s3).squeeze(1)#[1, 16, 64, 128])
        cost1_s3_possibility = F.softmax(cost1_s3, dim=1)#[1, 16, 64, 128])
        pred1_s3 = torch.sum(cost1_s3_possibility * disparity_samples_s3, dim=1, keepdim=True)#[1, 1, 64, 128])
        
        pred1_s3_cur = pred1_s3.detach()#[1, 1, 64, 128])
        pred1_v_s3 = disparity_variance_confidence(cost1_s3_possibility, disparity_samples_s3, pred1_s3_cur)#[1, 1, 64, 128])
        pred1_v_s3 = pred1_v_s3.sqrt()#[1, 1, 64, 128])
        mindisparity_s2 = pred1_s3_cur - (self.gamma_s2 + 1) * pred1_v_s3 - self.beta_s2#[1, 1, 64, 128])
        maxdisparity_s2 = pred1_s3_cur + (self.gamma_s2 + 1) * pred1_v_s3 + self.beta_s2#[1, 1, 64, 128])
        maxdisparity_s2 = F.upsample(maxdisparity_s2 * 2, [left.size()[2] // 2, left.size()[3] // 2], mode='bilinear',
                                     align_corners=True)#[1, 1, 128, 256])
        mindisparity_s2 = F.upsample(mindisparity_s2 * 2, [left.size()[2] // 2, left.size()[3] // 2], mode='bilinear',
                                     align_corners=True)#[1, 1, 128, 256])


        mindisparity_s2_1, maxdisparity_s2_1 = self.generate_search_range(self.sample_count_s2 + 1, mindisparity_s2, maxdisparity_s2, scale = 1)#([1, 1, 128, 256]) torch.Size([1, 1, 128, 256])
        disparity_samples_s2 = self.generate_disparity_samples(mindisparity_s2_1, maxdisparity_s2_1, self.sample_count_s2).float()#[1, 12, 128, 256])
        confidence_v_concat_s2, _ = self.cost_volume_generator(features_left["concat_feature2"],
                                                            features_right["concat_feature2"], disparity_samples_s2, 'concat')#[1, 12, 12, 128, 256])
        confidence_v_gwc_s2, disparity_samples_s2 = self.cost_volume_generator(features_left["gw2"], features_right["gw2"],
                                                                         disparity_samples_s2, 'gwc', self.num_groups // 2)#[1, 20, 12, 128, 256]) torch.Size([1, 1, 12, 128, 256])
        confidence_v_s2 = torch.cat((confidence_v_gwc_s2, confidence_v_concat_s2, disparity_samples_s2), dim=1)#[1, 33, 12, 128, 256])

        disparity_samples_s2 = torch.squeeze(disparity_samples_s2, dim=1)#[1, 12, 128, 256])

        cost0_s2 = self.confidence0_s2(confidence_v_s2)#[1, 16, 12, 128, 256])
        cost0_s2 = self.confidence1_s2(cost0_s2) + cost0_s2#[1, 16, 12, 128, 256])

        out1_s2 = self.confidence2_s2(cost0_s2)#[1, 16, 12, 128, 256])
        out2_s2 = self.confidence3_s2(out1_s2)#[1, 16, 12, 128, 256])

        cost1_s2 = self.confidence_classif1_s2(out2_s2).squeeze(1)#[1, 12, 128, 256])
        cost1_s2_possibility = F.softmax(cost1_s2, dim=1)#[1, 12, 128, 256])
        pred1_s2 = torch.sum(cost1_s2_possibility * disparity_samples_s2, dim=1, keepdim=True)#[1, 1, 128, 256])

        # pred1_v_s2 = disparity_variance_confidence(cost1_s2_possibility, disparity_samples_s2, pred1_s2)
        # pred1_v_s2 = pred1_v_s2.sqrt()

        if self.training:
            cost0_4 = self.classif0(cost0_4)#[1, 1, 32, 32, 64])
            cost1_4 = self.classif1(out1_4)#[1, 1, 32, 32, 64])

            cost0_4 = F.upsample(cost0_4, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear', align_corners=True)#[1, 1, 256, 256, 512])
            cost0_4 = torch.squeeze(cost0_4, 1)#[1, 256, 256, 512])
            pred0_4 = F.softmax(cost0_4, dim=1)#[1, 256, 256, 512])
            pred0_4 = disparity_regression(pred0_4, self.maxdisp)#[1, 256, 512])

            cost1_4 = F.upsample(cost1_4, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear', align_corners=True)#[1, 1, 256, 256, 512])
            cost1_4 = torch.squeeze(cost1_4, 1)#[1, 256, 256, 512])
            pred1_4 = F.softmax(cost1_4, dim=1)#[1, 256, 256, 512])
            pred1_4 = disparity_regression(pred1_4, self.maxdisp)#[1, 256, 512])

            pred2_s4 = F.upsample(pred2_s4 * 8, [left.size()[2], left.size()[3]], mode='bilinear', align_corners=True)#[1, 1, 256, 512])
            pred2_s4 = torch.squeeze(pred2_s4, 1)#[1, 256, 512])

            cost0_s3 = self.confidence_classif0_s3(cost0_s3).squeeze(1)#[1, 16, 64, 128])
            cost0_s3 = F.softmax(cost0_s3, dim=1)#[1, 16, 64, 128])
            pred0_s3 = torch.sum(cost0_s3 * disparity_samples_s3, dim=1, keepdim=True)#[1, 1, 64, 128])
            pred0_s3 = F.upsample(pred0_s3 * 4, [left.size()[2], left.size()[3]], mode='bilinear',
                                     align_corners=True)#[1, 1, 256, 512])
            pred0_s3 = torch.squeeze(pred0_s3, 1)#[1, 256, 512])

            costmid_s3 = self.confidence_classifmid_s3(out1_s3).squeeze(1)#([1, 16, 64, 128])
            costmid_s3 = F.softmax(costmid_s3, dim=1)#([1, 16, 64, 128])
            predmid_s3 = torch.sum(costmid_s3 * disparity_samples_s3, dim=1, keepdim=True)#[1, 1, 64, 128])
            predmid_s3 = F.upsample(predmid_s3 * 4, [left.size()[2], left.size()[3]], mode='bilinear',
                                     align_corners=True)#[1, 1, 256, 512])
            predmid_s3 = torch.squeeze(predmid_s3, 1)#[1, 256, 512]) 

            pred1_s3_up = F.upsample(pred1_s3 * 4, [left.size()[2], left.size()[3]], mode='bilinear', align_corners=True)#[1, 1, 256, 512])
            pred1_s3_up = torch.squeeze(pred1_s3_up, 1)#[1, 256, 512]) #############

            cost0_s2 = self.confidence_classif0_s2(cost0_s2).squeeze(1)#[1, 12, 128, 256])
            cost0_s2 = F.softmax(cost0_s2, dim=1)#[1, 12, 128, 256])
            pred0_s2 = torch.sum(cost0_s2 * disparity_samples_s2, dim=1, keepdim=True)#[1, 1, 128, 256])
            pred0_s2 = F.upsample(pred0_s2 * 2, [left.size()[2], left.size()[3]], mode='bilinear', align_corners=True)#[1, 1, 256, 512])
            pred0_s2 = torch.squeeze(pred0_s2, 1)#[1, 256, 512])

            costmid_s2 = self.confidence_classifmid_s2(out1_s2).squeeze(1)#[1, 12, 128, 256])
            costmid_s2 = F.softmax(costmid_s2, dim=1)#[1, 12, 128, 256])
            predmid_s2 = torch.sum(costmid_s2 * disparity_samples_s2, dim=1, keepdim=True)#[1, 1, 128, 256])
            predmid_s2 = F.upsample(predmid_s2 * 2, [left.size()[2], left.size()[3]], mode='bilinear',
                                     align_corners=True)#[1, 1, 256, 512])
            predmid_s2 = torch.squeeze(predmid_s2, 1)#[1, 256, 512])

            pred1_s2 = F.upsample(pred1_s2 * 2, [left.size()[2], left.size()[3]], mode='bilinear', align_corners=True)#[1, 1, 256, 512])
            pred1_s2 = torch.squeeze(pred1_s2, 1)#[1, 256, 512])

            return [pred0_4, pred1_4, pred2_s4, pred0_s3, predmid_s3, pred1_s3_up, pred0_s2, predmid_s2, pred1_s2]#

        else:
            pred2_s4 = F.upsample(pred2_s4 * 8, [left.size()[2], left.size()[3]], mode='bilinear', align_corners=True)
            pred2_s4 = torch.squeeze(pred2_s4, 1)

            pred1_s3_up = F.upsample(pred1_s3 * 4, [left.size()[2], left.size()[3]], mode='bilinear',
                                     align_corners=True)
            pred1_s3_up = torch.squeeze(pred1_s3_up, 1)

            pred1_s2 = F.upsample(pred1_s2 * 2, [left.size()[2], left.size()[3]], mode='bilinear', align_corners=True)
            pred1_s2 = torch.squeeze(pred1_s2, 1)


            return [pred1_s2], [pred1_s3_up], [pred2_s4]


def CFNet(d):
    return cfnet(d, use_concat_volume=True)