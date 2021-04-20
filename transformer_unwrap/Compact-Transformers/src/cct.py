import torch
import torch.nn as nn
import torch.nn.functional as F
from .transformers import TransformerEncoderLayer
# from transformers import TransformerEncoderLayer


__all__ = ['cct_2', 'cct_4', 'cct_6', 'cct_7', 'cct_8',
           'cct_10', 'cct_12', 'cct_24', 'cct_32',
           ]


        
class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=1,
            stride=2,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm2d(out_channels)

        super(Conv2dReLU, self).__init__(conv, bn, relu)
        
        
class CxnTokenizer(nn.Module):
    def __init__(self,
                 img_size,
                 stride=2,
                 kernel_size=3,
                 padding=1,
                 pooling_kernel_size=3,
                 pooling_stride=2, 
                 pooling_padding=1,
                 n_conv_layers=1,
                 activation=None,
                 max_pool=True,
                 use_batchnorm=True):
        super(CxnTokenizer, self).__init__()
        
        self.img_size = img_size
        self.conv1 = Conv2dReLU(3,4,kernel_size=3,padding=1,stride=stride,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv2dReLU(4,16,kernel_size=3,padding=1,stride=stride,
            use_batchnorm=use_batchnorm,
        )  # 把一个conv block集成了起来,加上了MaxPool2d,2倍的降采样
        self.conv3 = Conv2dReLU(16,64,kernel_size=3,padding=1,stride=stride,
            use_batchnorm=use_batchnorm,
        )
        self.conv4 = Conv2dReLU(64,256,kernel_size=3,padding=1,stride=stride,
            use_batchnorm=use_batchnorm,
        )
        self.flattener = nn.Flatten(2, 3) # 把第2维和第3维压平
        self.apply(self.init_weight)  # apply用于初始化

    def sequence_length(self, n_channels=3, height=512, width=512):
        a,b = self.forward(torch.zeros((1, n_channels, height, width)))
        return a.shape[1]

    def forward(self, x):
        '''
        把经过conv_layers的shape类似于[1,3,224,224]的图像展平
        用在我的模型中,这里也要改,我的模型为[1,3,512,512],这里要变为[256,32,32]
        '''
        output_feature=[]
        b = x[:,0,:,:]
        b0 = b.reshape([-1,4,int(self.img_size/2),int(self.img_size/2)])
        b1 = b.reshape([-1,16,int(self.img_size/4),int(self.img_size/4)])
        b2 = b.reshape([-1,64,int(self.img_size/8),int(self.img_size/8)])
        b3 = b.reshape([-1,256,int(self.img_size/16),int(self.img_size/16)])
        output_feature.append(b0)    # 保留初始的feature,就是wrap图
        output_feature.append(b1)
        output_feature.append(b2)
        output_feature.append(b3)
        x = self.conv1(x)  # [n,32,256,256]
        output_feature.append(x)
        x = self.conv2(x)  # [n,64,128,128]
        output_feature.append(x)
        x = self.conv3(x)  # [n,128,64,64]
        output_feature.append(x)
        x = self.conv4(x)  # [n,256,32,32] 保证这个与b一致
        output_feature.append(x)
        # 能不能return两个值还不一定,flattener输出[n, 512, 1024]
        output = self.flattener(x.transpose(-2, -1))
        return output,output_feature

    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)
            

class CXNTransformerUnet(nn.Module):
    def __init__(self, features,
                 seq_pool=True,
                 embedding_dim=1024,
                 num_layers=12,
                 num_heads=12,
                 mlp_ratio=4.0,
                 dropout_rate=0.1,
                 attention_dropout=0.1,
                 stochastic_depth_rate=0.1,
                 positional_embedding='sine',
                 sequence_length=None,
                 *args, **kwargs):
        super().__init__()
        # positional_embedding位置编码层只在encoder端和decoder端的embedding之后,
        # 第一个block之前出现
        positional_embedding = positional_embedding if \
            positional_embedding in ['sine', 'learnable', 'none'] else 'sine'
        dim_feedforward = int(embedding_dim * mlp_ratio)
        self.embedding_dim = embedding_dim
        self.sequence_length = sequence_length
        self.seq_pool = seq_pool

        assert sequence_length is not None or positional_embedding == 'none', \
            f"Positional embedding is set to {positional_embedding} and" \
            f" the sequence length was not specified."

        if not seq_pool:
            sequence_length += 1
            self.class_emb = nn.Parameter(torch.zeros(1, 1, self.embedding_dim),
                                          requires_grad=True) # 分类embedded?
        else:
            '''
            这里要注意,out的维度
            '''
            self.attention_pool = nn.Linear(self.embedding_dim, 256) # (in,out)

        if positional_embedding != 'none':  # 如果要进行positional_embedding
            if positional_embedding == 'learnable':
                self.positional_emb = nn.Parameter(
                    torch.zeros(1, sequence_length, embedding_dim),
                                                   requires_grad=True)
                nn.init.trunc_normal_(self.positional_emb, std=0.2)
                # 用从截断正态分布中提取的值填充输入张量
            else:
                self.positional_emb = nn.Parameter(
                    self.sinusoidal_embedding(sequence_length, embedding_dim),
                                                   requires_grad=False)
        else:
            self.positional_emb = None

        self.dropout = nn.Dropout(p=dropout_rate)
        dpr = [x.item() for x in torch.linspace(
            0, stochastic_depth_rate, num_layers)] # 0-0.1区间等分12份
        self.blocks = nn.ModuleList([
            TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads,
                                    dim_feedforward=dim_feedforward, dropout=dropout_rate,
                                    attention_dropout=attention_dropout, drop_path_rate=dpr[i])
            for i in range(num_layers)])
        # 把12个TransformerEncoderLayer编入subModule中,当作一个block
        self.norm = nn.LayerNorm(embedding_dim)

        '''
        注意,改动以下部分让其输出为512x512维度的结果
        '''
        self.upsampling = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv1 = Conv2dReLU(1024,1,kernel_size=3,padding=1,stride=1,
            use_batchnorm=False,)
        self.conv_up_768_64 = Conv2dReLU(768,64,kernel_size=3,padding=1,stride=1,
            use_batchnorm=False,)
        self.conv_up_192_16 = Conv2dReLU(192,16,kernel_size=3,padding=1,stride=1,
            use_batchnorm=False,)
        self.conv_up_48_4 = Conv2dReLU(48,4,kernel_size=3,padding=1,stride=1,
            use_batchnorm=False,)
        self.conv_up_12_1 = Conv2dReLU(12,1,kernel_size=3,padding=1,stride=1,
            use_batchnorm=False,)
        self.apply(self.init_weight)


    def forward(self, x,features):
        if self.positional_emb is None and x.size(1) < self.sequence_length:
            x = F.pad(x, (0, 0, 0, self.n_channels - x.size(1)), 
                      mode='constant', value=0)

        if not self.seq_pool:
            cls_token = self.class_emb.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)

        if self.positional_emb is not None:
            x += self.positional_emb

        x = self.dropout(x)

        for blk in self.blocks: # TransformerEncoderLayer
            x = blk(x)
        x = self.norm(x)        # 也还是保留原形状不变
        '''
        以上是transformer,接下来decoder
        '''

        if self.seq_pool:
            x = torch.matmul(F.softmax(
                self.attention_pool(x), dim=1).transpose(-1, -2), x)
        else:
            x = x[:, 0]
        
        x = x.reshape([-1,256,32,32])
        x = torch.cat((features[-1], x), dim=1)
        x = torch.cat((features[3], x), dim=1)
        x = self.upsampling(x) # [64,64]
        x = self.conv_up_768_64(x)
        
        x = torch.cat((features[-2], x), dim=1)  
        x = torch.cat((features[2], x), dim=1) 
        x = self.upsampling(x) # [128,128]
        x = self.conv_up_192_16(x)
        
        x = torch.cat((features[-3], x), dim=1) # 把原图拼接进去
        x = torch.cat((features[1], x), dim=1)
        x = self.upsampling(x) # [256,256]
        x = self.conv_up_48_4(x)
        
        x = torch.cat((features[-4], x), dim=1) # [n,64,256,256]
        x = torch.cat((features[0], x), dim=1)  # [n,4,256,256]
        x = self.upsampling(x) # [512,512]
        x = self.conv_up_12_1(x)       # [1,512,512]
        return x

    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @staticmethod
    def sinusoidal_embedding(n_channels, dim):
        '''
        用sin,cos进行位置编码
        '''
        pe = torch.FloatTensor([[p / (10000 ** (2 * (i // 2) / dim)) for i in range(dim)]
                                for p in range(n_channels)])
        pe[:, 0::2] = torch.sin(pe[:, 0::2])
        pe[:, 1::2] = torch.cos(pe[:, 1::2])
        return pe.unsqueeze(0)



class CXNCCT(nn.Module):
    def __init__(self,
                 img_size=512,
                 embedding_dim=1024,
                 n_input_channels=3,
                 n_conv_layers=1,
                 kernel_size=3,
                 stride=2,
                 padding=1,
                 pooling_kernel_size=3,
                 pooling_stride=2,
                 pooling_padding=1,
                 *args, **kwargs):
        super(CXNCCT, self).__init__()
        '''
        Tokenizer得输出3步降采样的features
        '''
        self.tokenizer = CxnTokenizer(img_size,
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   padding=padding,
                                   pooling_kernel_size=pooling_kernel_size,
                                   pooling_stride=pooling_stride,
                                   pooling_padding=pooling_padding,
                                   max_pool=True,
                                   activation=nn.ReLU,
                                   n_conv_layers=n_conv_layers)
        
        self.wrapUnet = CXNTransformerUnet(
            sequence_length=self.tokenizer.sequence_length(n_channels=n_input_channels,
                                                           height=img_size,
                                                           width=img_size),
            embedding_dim=embedding_dim,
            seq_pool=True,
            features=None,
            dropout=0.,
            attention_dropout=0.1,
            stochastic_depth=0.1,
            *args, **kwargs)

    def forward(self, x):
        x,features = self.tokenizer(x)
        return self.wrapUnet(x,features)


def _cxncct(num_layers, num_heads, mlp_ratio, embedding_dim,
         kernel_size=3, stride=None, padding=None,
         *args, **kwargs):
    stride = stride if stride is not None else max(1, (kernel_size // 2) - 1)
    padding = padding if padding is not None else max(1, (kernel_size // 2))
    return CXNCCT(num_layers=num_layers,
               num_heads=num_heads,
               mlp_ratio=mlp_ratio,
               embedding_dim=embedding_dim,
               kernel_size=kernel_size,
               stride=stride,
               padding=padding,
               *args, **kwargs)


def cct_2(*args, **kwargs):   # 这里注意embedding_dim
    return _cxncct(num_layers=2, num_heads=2, mlp_ratio=1, embedding_dim=1024,
                *args, **kwargs)


def cct_4(*args, **kwargs):
    return _cxncct(num_layers=4, num_heads=2, mlp_ratio=1, embedding_dim=128,
                *args, **kwargs)


def cct_6(*args, **kwargs):
    return _cxncct(num_layers=6, num_heads=4, mlp_ratio=2, embedding_dim=256,
                *args, **kwargs)


def cct_7(*args, **kwargs):
    return _cxncct(num_layers=7, num_heads=4, mlp_ratio=2, embedding_dim=256,
                *args, **kwargs)


def cct_8(*args, **kwargs):
    return _cxncct(num_layers=8, num_heads=4, mlp_ratio=2, embedding_dim=256,
                *args, **kwargs)


def cct_10(*args, **kwargs):
    return _cxncct(num_layers=10, num_heads=8, mlp_ratio=3, embedding_dim=512,
                *args, **kwargs)


def cct_12(*args, **kwargs):
    return _cxncct(num_layers=12, num_heads=12, mlp_ratio=4, embedding_dim=768,
                *args, **kwargs)


def cct_24(*args, **kwargs):
    return _cxncct(num_layers=24, num_heads=16, mlp_ratio=4, embedding_dim=1024,
                *args, **kwargs)


def cct_32(*args, **kwargs):
    return _cxncct(num_layers=32, num_heads=16, mlp_ratio=4, embedding_dim=1280,
                *args, **kwargs)



