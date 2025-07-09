import random
import cv2
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

######################################################################
class GeM(nn.Module):
    # GeM zhedong zheng
    def __init__(self, dim = 2048, p=3, eps=1e-6):
        super(GeM,  self).__init__()
        self.p = nn.Parameter(torch.ones(dim)*p, requires_grad = True) # initial p
        self.eps = eps
        self.dim = dim
    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        x = torch.transpose(x, 1, -1)
        x = x.clamp(min=eps).pow(p)
        x = torch.transpose(x, 1, -1)
        x = F.avg_pool2d(x, (x.size(-2), x.size(-1)))
        x = x.view(x.size(0), x.size(1))
        x = x.pow(1./p)
        return x

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ',' + 'dim='+str(self.dim)+')'

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in') # For old pytorch, you may use kaiming_normal.
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)

def fix_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace = True

# def slice(x, ratio):
#     batch_size, channels, height, width = x.size()
#     center_height = int(height * ratio)
#     center_width = int(width * ratio)
#     start_height = (height-center_height)//2
#     start_width = (width-center_width)//2
#     # start_height = random.randint(1, 11)
#     # start_width = random.randint(1, 11)
#     center = x[:, :, start_height:start_height+center_height, start_width:start_width+center_width]
#     return center, x[:, :, :, :start_width], x[:, :, :, start_width+center_width:], x[:, :, :start_height, start_width:start_width+center_width], x[:, :, start_height+center_height:, start_width:start_width+center_width]

# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|
class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, droprate, relu=False, bnorm=True, num_bottleneck=512, linear=True, return_f = False):
        super(ClassBlock, self).__init__()
        self.return_f = return_f
        add_block = []
        if linear:
            add_block += [nn.Linear(input_dim, num_bottleneck)]
        else:
            num_bottleneck = input_dim
        if bnorm:
            add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if droprate > 0:
            add_block += [nn.Dropout(p=droprate)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier
    def forward(self, x):
        x = self.add_block(x)
        if self.return_f:
            f = x
            x = self.classifier(x)
            return x, f
        else:
            x = self.classifier(x)
            return x
class ChannelAttention(nn.Module):
    def __init__(self, num_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(num_channels, num_channels // reduction_ratio, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(num_channels // reduction_ratio, num_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class GAM(nn.Module):
    def __init__(self, in_channels, out_channels, rate=4):
        super(GAM, self).__init__()
        self.channel_attention = nn.Sequential(
            nn.Linear(in_channels, in_channels//rate),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels//rate, in_channels)
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        b, c, h, w = x.size()
        channel_att = torch.mean(x, dim=[2, 3])
        channel_att = self.channel_attention(channel_att)
        channel_att = torch.sigmoid(channel_att).unsqueeze(2).unsqueeze(3).expand_as(x)
        spatial_att = self.spatial_attention(x)
        spatial_att = torch.sigmoid(spatial_att)
        out = x * channel_att * spatial_att
        return out

class mambaout(nn.Module):
    def __init__(self, in_channels, kernel_size=7, conv_ratio=1):
        super(mambaout, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.conv_ratio = conv_ratio
        self.conv_channels = int(in_channels*conv_ratio)
        self.depthwise_conv = nn.Conv2d(
            self.conv_channels, self.conv_channels,
            kernel_size=kernel_size, padding=kernel_size//2,
            groups=self.conv_channels
        )
        self.bn = nn.BatchNorm2d(self.conv_channels)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x_conv = x[:, :self.conv_channels, :, :]
        x_conv = self.depthwise_conv(x_conv)
        x_conv = self.bn(x_conv)
        x_conv = self.sigmoid(x_conv)
        out = x*x_conv
        return out

class CBAM(nn.Module):
    def __init__(self, num_channels, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(num_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x

class SEattention(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEattention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(channels, channels // reduction, bias=False),
                                nn.ReLU(inplace=True),
                                nn.Linear(channels // reduction, channels, bias=False),
                                nn.Sigmoid()
                                )
    def forward(self,x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# Define the ResNet50-based Model
class ft_net(nn.Module):
    def __init__(self, droprate=0.5, stride=1, init_model=None, pool='gem'):
        super(ft_net, self).__init__()

        model_ft = models.resnet50(pretrained=True)  # 预训练模型
        model_ft.avgpool = nn.Sequential()
        model_ft.fc = nn.Sequential()  # remove fc

        # avg pooling to global pooling
        if stride == 1:
            model_ft.layer4[0].downsample[0].stride = (1, 1)
            model_ft.layer4[0].conv2.stride = (1, 1)

        # self.pool = pool
        # if pool =='avg+max':
        #     model_ft.avgpool2 = nn.AdaptiveAvgPool2d((1, 1))
        #     model_ft.maxpool2 = nn.AdaptiveMaxPool2d((1, 1))
        #     # self.classifier = ClassBlock(4096, class_num, droprate)
        # elif pool == 'avg':
        #     model_ft.avgpool2 = nn.AdaptiveAvgPool2d((1, 1))
        #     # self.classifier = ClassBlock(2048, class_num, droprate)
        # elif pool == 'max':
        #     model_ft.maxpool2 = nn.AdaptiveMaxPool2d((1, 1))
        # elif pool == 'gem':
        #     model_ft.gem2 = GeM(dim=2048)

        self.model = model_ft
        # self.cbam1 = CBAM(256)  # 对应ResNet的layer4输出通道数
        # self.cbam2 = CBAM(512)
        # self.cbam3 = CBAM(1024)
        # self.cbam4 = CBAM(2048)
        self.attention1 = SEattention(256)
        self.attention2 = SEattention(512)
        self.attention3 = SEattention(1024)
        self.attention4 = SEattention(2048)
        # self.mamba1 = mambaout(in_channels=256)
        # self.mamba2 = mambaout(in_channels=512)
        # self.mamba3 = mambaout(in_channels=1024)
        # self.mamba4 = mambaout(in_channels=2048)
        # self.gam1 = GAM(in_channels=256, out_channels=256)
        # self.gam2 = GAM(in_channels=512, out_channels=512)
        # self.gam3 = GAM(in_channels=1024, out_channels=1024)
        # self.gam4 = GAM(in_channels=2048, out_channels=2048)

        if init_model != None:
            self.model = init_model.model
            self.pool = init_model.pool
            # self.classifier.add_block = init_model.classifier.add_block

    def forward(self, x):  # 前向传播
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        # x = self.gam1(x)
        x = self.attention1(x)
        # x = self.mamba1(x)
        # x = self.cbam1(x)
        x = self.model.layer2(x)
        # x = self.gam2(x)
        x = self.attention2(x)
        # x = self.mamba2(x)
        # x = self.cbam2(x)
        x = self.model.layer3(x)
        # x = self.gam3(x)
        # x = self.mamba3(x)
        x = self.attention3(x)
        # x = self.cbam3(x)
        x = self.model.layer4(x)
        # x = self.gam4(x)
        x = self.attention4(x)
        # x = self.mamba4(x)
        # x = self.cbam4(x)

        # if self.pool == 'avg+max':
        #     x1 = self.model.avgpool2(x)
        #     x2 = self.model.maxpool2(x)
        #     x = torch.cat((x1, x2), dim=1)
        # elif self.pool == 'avg':
        #     x = self.model.avgpool2(x)
        # elif self.pool == 'max':
        #     x = self.model.maxpool2(x)
        # elif self.pool == 'gem':
        #     x = self.model.gem2(x)

        # x = x.view(x.size(0), x.size(1))  # 把x转换为[batchsize,chnnels]的矩阵
        # x = self.classifier(x)

        return x


class two_view_net(nn.Module):
    def __init__(self, class_num, droprate=0.5, stride=2, pool='avg', share_weight=False, VGG16 = False, circle=False, ratio=0.5):
        super(two_view_net, self).__init__()
        # if VGG16:
        #     self.model_1 = ft_net_VGG16(class_num, stride=stride, pool=pool)
        # else:
        self.model_1 = ft_net(class_num, stride=stride, pool=pool)   ##
        if share_weight:
            self.model_2 = self.model_1  ##
        else:
            # if VGG16:
            #     self.model_2 = ft_net_VGG16(class_num, stride = stride, pool = pool)
            # else:
            self.model_2 = ft_net(class_num, stride=stride, pool=pool)

        self.circle = circle  ##

        self.classifier_1 = ClassBlock(8192, class_num, droprate, bnorm=True, linear=True, return_f=circle)
        self.classifier_2 = ClassBlock(2048, class_num, droprate, bnorm=True, linear=True, return_f=circle)
        # self.classifier = ClassBlock(2048, class_num, droprate, return_f = circle)  ##
        # if pool =='avg+max':
        #     self.classifier = ClassBlock(4096, class_num, droprate, return_f = circle)
        # if VGG16:
        #     self.classifier = ClassBlock(512, class_num, droprate, return_f = circle)
        #     if pool =='avg+max':
        #         self.classifier = ClassBlock(1024, class_num, droprate, return_f = circle)

        self.pool = pool
        if pool == 'avg+max':
            self.model_1.avgpool2 = nn.AdaptiveAvgPool2d((1, 1))
            self.model_1.maxpool2 = nn.AdaptiveMaxPool2d((1, 1))
            # self.classifier = ClassBlock(4096, class_num, droprate)
        elif pool == 'avg':
            self.model_1.avgpool2 = nn.AdaptiveAvgPool2d((1, 1))
            # self.classifier = ClassBlock(2048, class_num, droprate)
        elif pool == 'max':
            self.model_1.maxpool2 = nn.AdaptiveMaxPool2d((1, 1))
        elif pool == 'gem':
            self.model_1.model.gem2 = GeM(dim=2048)

        self.ratio = ratio

    # def slice(self, x):
    #     batch_size, channels, height, width = x.size()
    #     center_height = int(height * self.ratio)
    #     center_width = int(width * self.ratio)
    #     start_height = (height - center_height) // 2
    #     start_width = (width - center_width) // 2
    #     center = x[:, :, start_height:start_height + center_height, start_width:start_width + center_width]
    #     return center, x[:, :, :, :start_width], x[:, :, :, start_width + center_width:], x[:, :, :start_height, start_width:start_width + center_width], x[:, :, start_height + center_height:, start_width:start_width + center_width]

    def slice(self, x, high_value_indices):
        batch_size, channels, height, width = x.size()
        center_height = int(height * self.ratio)
        center_width = int(width * self.ratio)
        start_height = (height - center_height) // 2
        start_width = (width - center_width) // 2
        top_left_row = np.min(high_value_indices[1], initial=6)
        top_left_col = np.min(high_value_indices[0], initial=6)
        # bottom_right_row = np.max(high_value_indices[1], initial=18)
        # bottom_right_col = np.max(high_value_indices[0], initial=18)
        # print(top_left_col)
        # print(bottom_right_col)
        # print(top_left_row)
        # print(bottom_right_row)
        # if bottom_right_col in range(top_left_col+1, 18) and bottom_right_row in range(top_left_row+1, 18):
            # center = x[:, :, top_left_row:bottom_right_row, top_left_col:bottom_right_col]
            # left = x[:, :, :, :top_left_col]
            # right = x[:, :, :, bottom_right_col:]
            # top = x[:, :, :top_left_row, top_left_col:bottom_right_col]
            # bottom = x[:, :, bottom_right_row:, top_left_col:bottom_right_col]
        if top_left_col+center_width < 23 and top_left_row+center_height < 23:
            center = x[:, :, top_left_row:top_left_row+center_height, top_left_col:top_left_col+center_width]
            left = x[:, :, :, :top_left_col]
            right = x[:, :, :, top_left_col + center_width:]
            top = x[:, :, :top_left_row, top_left_col:top_left_col+center_width]
            bottom = x[:, :, top_left_row+center_height:, top_left_col:top_left_col+center_width]
        else:
            center = x[:, :, start_height:start_height+center_height, start_width:start_width+center_width]
            left = x[:, :, :, :start_width]
            right = x[:, :, :, start_width + center_width:]
            top = x[:, :, :start_height, start_width:start_width + center_width]
            bottom = x[:, :, start_height + center_height:, start_width:start_width + center_width]
        return center, left, right, top, bottom

    def generate_heatmap(self, feature_tensor):
        # 计算每个特征图的平均值
        heatmap = torch.mean(feature_tensor, dim=1)  # 结果形状为[8, 24, 24]
        # 归一化热力图
        max_value = torch.max(heatmap)
        min_value = torch.min(heatmap)
        heatmap = (heatmap - min_value) / (max_value - min_value) * 255  # 归一化到0-255
        heatmap = heatmap.cpu().detach().numpy().astype(np.uint8)  # 转换为numpy数组
        # 将单通道热力图转换为三通道
        heatmaps = []
        for i in range(heatmap.shape[0]):
            heatmap_single = cv2.applyColorMap(heatmap[i], cv2.COLORMAP_JET) # 使用JET颜色映射
            heatmap_single = cv2.cvtColor(heatmap_single, cv2.COLOR_BGR2RGB)  # 转换颜色空间为RGB
            heatmaps.append(heatmap_single)
        return heatmaps

    def forward(self, x1, x2):
        if x1 is None:
            y1_remain = None
            y1_center = None
            y1 = None
        else:
            x1 = self.model_1(x1)
            heatmaps = self.generate_heatmap(x1)
            mean_heatmap = np.mean(heatmaps, axis=0)
            mean_heatmap = cv2.cvtColor(mean_heatmap.astype(np.uint8), cv2.COLOR_BGR2RGB)
            data = mean_heatmap.reshape(-1, 3)
            pca = PCA(n_components=2)
            principalComponents = pca.fit_transform(data)
            pca_heatmap = principalComponents[:, 0].reshape(24, 24)
            pca_heatmap = (pca_heatmap-pca_heatmap.min()) / (pca_heatmap.max()-pca_heatmap.min())
            # max_value = np.unravel_index(np.argmax(pca_heatmap), pca_heatmap.shape)
            threshold = np.percentile(pca_heatmap, 15)
            high_value = pca_heatmap < threshold
            high_value[0:4, :] = False
            high_value[19:24, :] = False
            high_value[4:19, 0:4] = False
            high_value[4:19, 19:24] = False
            high_value_indices = np.where(high_value)
            # print(high_value_indices)
            # high_value_points = pca_heatmap[high_value_indices]
            x1_center, x1_left, x1_right, x1_top, x1_bottom = self.slice(x1, high_value_indices)
            # fig, axs = plt.subplots(1, 2, figsize=(12, 6))
            # axs[0].imshow(mean_heatmap, cmap='jet')
            # axs[0].set_title('original')
            # axs[0].axis('off')
            # axs[1].imshow(pca_heatmap, cmap='jet')
            # axs[1].set_title('PCA')
            # axs[1].axis('off')
            # plt.imshow(pca_heatmap, cmap="jet")
            # plt.tight_layout()
            # plt.show()
            # plt.imsave('mean_heatmap.png', mean_heatmap)
            # plt.subplot(1, 2, 2)
            # plt.imshow(pca_heatmap, cmap='hot', interpolation='nearest', alpha=0.5)
            # plt.scatter(high_value_indices[0], high_value_indices[1], c=high_value_points, cmap='hot', s=100, edgecolors='none')
            # plt.colorbar()
            # plt.title('high values')
            # plt.show()
            # plt.imsave('pca_heatmap.png', pca_heatmap)

            # x1_center, x1_left, x1_right, x1_top, x1_bottom = self.slice(x1)
            x1_center = self.model_1.model.gem2(x1_center)
            y1_center = self.classifier_2(x1_center)
            x1_left = self.model_1.model.gem2(x1_left)
            x1_right = self.model_1.model.gem2(x1_right)
            x1_bottom = self.model_1.model.gem2(x1_bottom)
            x1_top = self.model_1.model.gem2(x1_top)
            x3 = torch.cat([x1_left, x1_bottom, x1_right, x1_top], dim=1)
            y1_remain = self.classifier_1(x3)
            x1 = self.model_1.model.gem2(x1)
            y1 = self.classifier_2(x1)

        if x2 is None:
            y2_remain = None
            y2_center = None
            y2 = None
        else:
            x2 = self.model_2(x2)
            heatmaps = self.generate_heatmap(x2)
            mean_heatmap = np.mean(heatmaps, axis=0)
            mean_heatmap = cv2.cvtColor(mean_heatmap.astype(np.uint8), cv2.COLOR_BGR2RGB)
            data = mean_heatmap.reshape(-1, 3)
            pca = PCA(n_components=2)
            principalComponents = pca.fit_transform(data)
            pca_heatmap = principalComponents[:, 0].reshape(24, 24)
            pca_heatmap = (pca_heatmap - pca_heatmap.min()) / (pca_heatmap.max() - pca_heatmap.min())
            # max_value = np.unravel_index(np.argmax(pca_heatmap), pca_heatmap.shape)
            threshold = np.percentile(pca_heatmap, 15)
            high_value = pca_heatmap < threshold
            high_value[0:4, :] = False
            high_value[19:24, :] = False
            high_value[4:19, 0:4] = False
            high_value[4:19, 19:24] = False
            high_value_indices = np.where(high_value)
            # fig, axs = plt.subplots(1, 2, figsize=(12, 6))
            # axs[0].imshow(mean_heatmap, cmap='jet')
            # axs[0].set_title('original')
            # axs[0].axis('off')
            # axs[1].imshow(pca_heatmap, cmap='jet')
            # axs[1].set_title('PCA')
            # axs[1].axis('off')
            # plt.imshow(pca_heatmap, cmap="jet")
            # plt.tight_layout()
            # plt.show()
            x2_center, x2_left, x2_right, x2_top, x2_bottom = self.slice(x2, high_value_indices)
            # x2_center, x2_left, x2_right, x2_top, x2_bottom = self.slice(x2)
            x2_center = self.model_1.model.gem2(x2_center)
            y2_center = self.classifier_2(x2_center)
            x2_right = self.model_1.model.gem2(x2_right)
            x2_left = self.model_1.model.gem2(x2_left)
            x2_bottom = self.model_1.model.gem2(x2_bottom)
            x2_top = self.model_1.model.gem2(x2_top)
            x4 = torch.cat([x2_left, x2_bottom, x2_right, x2_top], dim=1)
            y2_remain = self.classifier_1(x4)
            x2 = self.model_1.model.gem2(x2)
            y2 = self.classifier_2(x2)

        return y1, y2, y1_remain, y1_center, y2_remain, y2_center


class three_view_net(nn.Module):
    def __init__(self, class_num, droprate, stride = 2, pool = 'avg', share_weight = False, VGG16=False, circle=False):
        super(three_view_net, self).__init__()
        if VGG16:
            self.model_1 = ft_net_VGG16(class_num, stride = stride, pool = pool)
            self.model_2 = ft_net_VGG16(class_num, stride = stride, pool = pool)
        else:
            self.model_1 = ft_net(class_num, stride = stride, pool = pool)
            self.model_2 = ft_net(class_num, stride = stride, pool = pool)

        if share_weight:
            self.model_3 = self.model_1
        else:
            if VGG16:
                self.model_3 = ft_net_VGG16(class_num, stride = stride, pool = pool)
            else:
                self.model_3 = ft_net(class_num, stride = stride, pool = pool)

        self.circle = circle

        self.classifier = ClassBlock(2048, class_num, droprate, return_f = circle)  # 分类器

        if pool =='avg+max':
            self.classifier = ClassBlock(4096, class_num, droprate, return_f = circle)

    def forward(self, x1, x2, x3, x4 = None): # x4 is extra data
        if x1 is None:
            y1 = None
        else:
            x1 = self.model_1(x1)
            x1 = x1.view(x1.size(0), x1.size(1))
            y1 = self.classifier(x1)

        if x2 is None:
            y2 = None
        else:
            x2 = self.model_2(x2)
            x2 = x2.view(x2.size(0), x2.size(1))
            y2 = self.classifier(x2)

        if x3 is None:
            y3 = None
        else:
            x3 = self.model_3(x3)
            x3 = x3.view(x3.size(0), x3.size(1))
            y3 = self.classifier(x3)

        if x4 is None:
            return y1, y2, y3
        else:
            x4 = self.model_2(x4)
            x4 = x4.view(x4.size(0), x4.size(1))
            y4 = self.classifier(x4)
            return y1, y2, y3, y4


'''
# debug model structure
# Run this code with:
python model.py
'''
if __name__ == '__main__':
# Here I left a simple forward function.
# Test the model, before you train it.
#     net = ft_net()
#     # net = two_view_net(751, droprate=0.5, VGG16=True)
#     # net.classifier = nn.Sequential()
#     print(net)
#     input = Variable(torch.FloatTensor(8, 3, 256, 256))
#     output = net(input)
#     print('net output size:')
#     print(output.shape)
    net = two_view_net(701, droprate=0.5, VGG16=True)
    # net.classifier = nn.Sequential()
    print(net)
    input = Variable(torch.FloatTensor(8, 3, 256, 256))
    output, output2 = net(input, input)
    print('net output size:')
    print(output.shape)
    print(output2.shape)




