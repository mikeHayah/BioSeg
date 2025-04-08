import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

def init_weights(net, init_type='kaiming', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)
    print('initialize network with %s' % init_type)
    net.apply(init_func)

class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2)
        )


    def forward(self,x):
        x = self.conv(x)
        return x

class upsample_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(upsample_conv,self).__init__()
        self.up = nn.Sequential(
            #nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            #nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1),
            nn.ConvTranspose2d(ch_in, ch_out, kernel_size=2, stride=2),
            nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x

class Attention_block(nn.Module):
    def __init__(self,F_g,F_x,F_int):
        super(Attention_block,self).__init__()
        streach = 1024//F_int
        self.W_g = nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True)
        self.N_g = nn.LayerNorm([F_int, 8*streach, 8*streach])
        self.W_x = nn.Conv2d(F_x, F_int, kernel_size=1,stride=1,padding=0,bias=True)
        self.N_x = nn.LayerNorm([F_int, 8*streach, 8*streach])
        self.relu = nn.ReLU(inplace=True)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.Sigmoid(),
            #nn.Dropout2d(0.2)
        )
        self.match_ch = nn.Sequential(
            nn.Conv2d(F_x, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
            #nn.Softmax2d()
            nn.ReLU(inplace=True)
            )
        
    def forward(self,g,x):
        g1 = self.W_g(g)
        g1 = self.N_g(g1)
        x1 = self.W_x(x)
        if g1.size() != x1.size():
            #l1 = F.interpolate(l1, size=g1.shape[2:])
            x1 = F.interpolate(x1, size=g1.shape[2:], mode='bilinear', align_corners=False)
        x1 = self.N_x(x1)
        z = self.relu(g1+x1)
        a = self.psi(z)
        
        if a.size() != x.size():
            a = F.interpolate(a, size=x.shape[2:], mode='bilinear', align_corners=False)
        a = a / (a.sum(dim=(2, 3), keepdim=True) + 1e-6)
        x_att = a*x
        x_adjusted = self.match_ch(x_att)
        return x_adjusted #, a

class QKV_attention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(QKV_attention, self).__init__()
        self.query = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        streach = 2048//in_channels
        self.key = nn.ConvTranspose2d(2048, out_channels, kernel_size=streach, stride=streach, padding=0)
        self.value = nn.Sequential(
            nn.ConvTranspose2d(2048, out_channels, kernel_size=streach, stride=streach, padding=0),
            nn.LayerNorm([out_channels, 8*streach, 8*streach]))                 # =8 for image patch size 256 /32  ,or  = 16 for image patch size 512 /32
        self.gamma = nn.Parameter(torch.zeros(1))
        self.normalize = nn.LayerNorm([out_channels, 8*streach, 8*streach])
        
    def forward(self, q, k):
        batch_size, C, H, W = q.size()
        query = self.query(q)
        query = query.view(batch_size, -1, H*W).permute(0, 2, 1)
        key = self.key(k)
        if key.size() != q.size():
            key = F.interpolate(key, size=q.shape[2:], mode='bilinear', align_corners=False)
        key = key.view(batch_size, -1, H*W)
        value = self.value(k)
        if value.size() != q.size():
            value = F.interpolate(value, size=q.shape[2:], mode='bilinear', align_corners=False)
        value = value.view(batch_size, -1, H*W).permute(0, 2, 1)
        
        raw_scores = torch.bmm(query, key)
        d_k = query.size(-1)
        scaled_scores = raw_scores / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
        attention_weight = F.softmax(scaled_scores, dim=-1)
        
        attention_out = torch.bmm(attention_weight, value).view(batch_size, H, W, C).permute(0, 3, 1, 2)        
        out = self.normalize(self.gamma * attention_out + q) #x
        return out

class kernel_branching(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=[1, 3, 5, 7]):
        super(kernel_branching, self).__init__()
        self.branches = nn.ModuleList([
			nn.Sequential(
				nn.Conv2d(in_channels, out_channels, kernel_size=ks, padding=ks//2, stride=1),
				nn.BatchNorm2d(out_channels),  # Batch normalization
				nn.ReLU(inplace=True)
			) for ks in kernel_sizes  # Different kernel sizes
		])

    def forward(self, x):
        return [branch(x) for branch in self.branches]

class U_Net_plus(nn.Module):
    def __init__(self,img_ch=1,output_ch=1):
        super(U_Net_plus,self).__init__()
        
        self.normalizer = nn.LayerNorm([1, 256, 256])

        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=64)
        self.Conv2 = conv_block(ch_in=64,ch_out=128)
        self.Conv3 = conv_block(ch_in=128,ch_out=256)
        self.Conv4 = conv_block(ch_in=256,ch_out=512)
        self.Conv5 = conv_block(ch_in=512,ch_out=1024)
        self.Conv6 = conv_block(ch_in=1024,ch_out=2048)

        self.btnk_ch = 2048
        kernel_sizes=[1, 3, 5, 7]
        self.kernel_branches = kernel_branching(self.btnk_ch, self.btnk_ch, kernel_sizes)


        self.Upsample_conv5 = upsample_conv(ch_in=2048, ch_out=1024)
        self.QKVatt5 = QKV_attention(in_channels=1024, out_channels=1024)
        self.Att5 = Attention_block(F_g=1024,F_x=1024,F_int=512)
        self.conv_up5 = conv_block(ch_in=1024, ch_out=512)
        self.norm5 = nn.LayerNorm([1024, 16, 16])
        
        self.Upsample_conv4 = upsample_conv(ch_in=1024, ch_out=512)
        self.QKVatt4 = QKV_attention(in_channels=512, out_channels=512)
        self.Att4 = Attention_block(F_g=512,F_x=512,F_int=256)
        self.conv_up4 = conv_block(ch_in=512, ch_out=256)
        self.norm4 = nn.LayerNorm([512, 32, 32])
        
        self.Upsample_conv3 = upsample_conv(ch_in=512, ch_out=256)
        self.QKVatt3 = QKV_attention(in_channels=256, out_channels=256)
        self.Att3 = Attention_block(F_g=256,F_x=256,F_int=128)
        self.conv_up3 = conv_block(ch_in=256, ch_out=128)
        self.norm3 = nn.LayerNorm([256, 64, 64])


        self.Upsample_conv2 = upsample_conv(ch_in=256, ch_out=128)
        self.Att2 = Attention_block(F_g=128,F_x=128,F_int=64)
        self.conv_up2 = conv_block(ch_in=128, ch_out=64)
        self.norm2 = nn.LayerNorm([128, 128, 128])

        self.Upsample_conv1 = upsample_conv(ch_in=128, ch_out=64)
        self.Att1 = Attention_block(F_g=64,F_x=64,F_int=32)
        self.conv_up1 = conv_block(ch_in=64, ch_out=32)
        self.norm1 = nn.LayerNorm([64, 256, 256])


        self.Conv_1x1 = nn.Conv2d(64,output_ch,kernel_size=1,stride=1,padding=0)

        self.projection6 = nn.Conv2d(2048, 1, kernel_size=1)                   
        self.projection5 = nn.Conv2d(1024, 1, kernel_size=1) 
        self.projection4 = nn.Conv2d(512, 1, kernel_size=1) 
        self.projection3 = nn.Conv2d(256, 1, kernel_size=1) 
        self.projection2 = nn.Conv2d(128, 1, kernel_size=1) 
         
    def forward(self,x):
        # Normalization
        x = self.normalizer(x)      
        
        # encoding path
        x1 = self.Conv1(x)                  # 1x 64
        x2 = self.Maxpool(x1)               
        x2 = self.Conv2(x2)                 # 2x 128
        x3 = self.Maxpool(x2)                  
        x3 = self.Conv3(x3)                 # 4x 256
        x4 = self.Maxpool(x3)               
        x4 = self.Conv4(x4)                 # 8x 512
        x5 = self.Maxpool(x4)               
        x5 = self.Conv5(x5)                 # 16x 1064
        x6 = self.Maxpool(x5)               
        x6 = self.Conv6(x6)                 # 32x 2048

        # bottleneck     
        [k1, k3, k5, k7] = self.kernel_branches(x6)     # 32x 2048
        
        # decoding + concat path
        q5 = self.Upsample_conv5(k1)         # 16x 1064
        g5 =  self.QKVatt5(q5, k3)           # 16x 1024
        a5 = self.Att5(g=g5,x=x5)            # 16x 512 
        d5 = self.conv_up5(g5)               # 16x 512
        d5 = torch.nn.functional.interpolate(d5, size=a5.shape[2:]) # 16x 512
        c5 = torch.cat((a5,d5),dim=1)        # 16x 1024
        c5 = self.norm5(c5)

        q4 = self.Upsample_conv4(c5)         # 8x 512
        g4 = self.QKVatt4(q4, k5)            # 8x 512
        a4 = self.Att4(g=g4,x=x4)            # 8x 256
        d4 = self.conv_up4(g4)               # 8x 256
        d4 = torch.nn.functional.interpolate(d4, size=a4.shape[2:])     # 8x 256
        c4 = torch.cat((a4,d4),dim=1)        # 8x 512
        c4 = self.norm4(c4)

        q3 = self.Upsample_conv3(c4)         # 4x 256
        g3 = self.QKVatt3(q3, k7)            # 4x 256   
        a3 = self.Att3(g=g3,x=x3)            # 4x 128
        d3 = self.conv_up3(g3)               # 4x 128
        d3 = torch.nn.functional.interpolate(d3, size=a3.shape[2:])     # 4x 128
        c3 = torch.cat((a3,d3),dim=1)        # 4x 256
        c3 = self.norm3(c3)

        g2 = self.Upsample_conv2(c3)         # 2x 128   
        a2 = self.Att2(g=g2,x=x2)            # 2x 64
        d2 = self.conv_up2(g2)               # 2x 64
        d2 = torch.nn.functional.interpolate(d2, size=a2.shape[2:])     # 2x 64
        c2 = torch.cat((a2,d2),dim=1)        # 2x 128
        c2 = self.norm2(c2)

        g1 = self.Upsample_conv1(c2)        # 1x 64
        a1 = self.Att1(g=g1,x=x1)           # 1x 32
        d1 = self.conv_up1(g1)              # 1x 32
        d1 = torch.nn.functional.interpolate(d1, size=a1.shape[2:])     # 1x 32
        c1 = torch.cat((a1,d1),dim=1)        # 1x 64
        c1 = self.norm1(c1)

        d = self.Conv_1x1(c1)               # 1x 1

        c6 = self.projection6(x6)    # model 10-2-2025
        c5 = self.projection5(c5)
        c4 = self.projection4(c4)
        c3 = self.projection3(c3)
        c2 = self.projection2(c2)
        img_decoder = [c6, c5, c4, c3, c2]

        return d, img_decoder


