import torch
import torch.nn as nn
import torch.nn.functional as F
from .transformers import TransformerEncoderLayer
# from transformers import TransformerEncoderLayer


__all__ = ['cct_2', 'cct_4', 'cct_6', 'cct_7', 'cct_8',
           'cct_10', 'cct_12','cot_2', 'cot_4', 'cot_6', 'cot_7', 'cot_8',
           'cot_10', 
           ]


# more output      
class Conv2dReLU(nn.Sequential):
    '''
    有maxpooling
    '''
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=1,
            stride=1,
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
        relu = nn.LeakyReLU(0.1,inplace=True)
        
        avgpool = nn.AvgPool2d((2, 2), stride=(2, 2))

        bn = nn.BatchNorm2d(out_channels)

        super(Conv2dReLU, self).__init__(conv, bn, relu,avgpool)
        
        
class Conv2dReLUNoPooling(nn.Sequential):
    '''
    无avgpooling
    '''
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=1,
            stride=1,
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
        relu = nn.LeakyReLU(0.1,inplace=True)

        bn = nn.BatchNorm2d(out_channels)

        super(Conv2dReLUNoPooling, self).__init__(conv, bn, relu)
        
        
class Conv2dFinal(nn.Sequential):
    '''
    最后一层不加relu,因为要预测负值
    '''
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=1,
            stride=1,
            use_batchnorm=False,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        #avgpool = nn.AvgPool2d((2, 2), stride=(1, 1))   # 最后加一个avgpool
        super(Conv2dFinal, self).__init__(conv)
        
        
class Conv2dTranpose(nn.Sequential):
    '''
    反卷积,代替upsampling插值
    '''
    def __init__(
            self,
            in_channels=1, 
        	out_channels=1, 
        	stride=1, 
        	kernel_size=3, 
        	padding=1, 
        	output_padding=0,
        	dilation=1, 
        	padding_mode="zeros", 
        	bias=False,

    ):
        conv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            dilation = dilation,
            bias=not (bias),
        )
        super(Conv2dTranpose, self).__init__(conv)
        
        
class OnlyTransformerToken(nn.Module):
    '''
    仅仅采用transformer的模型,input[n,3,256,256],output[n,1,256,256]
    '''
    def __init__(self,img_size,
                 seq_pool=True,
                 embedding_dim=256,
                 num_layers=12,
                 num_heads=12,
                 mlp_ratio=4.0,
                 dropout_rate=0.1,
                 attention_dropout=0.1,
                 stochastic_depth_rate=0.1,
                 positional_embedding='sine',
                 sequence_length=256,
                 *args, **kwargs):
        super(OnlyTransformerToken, self).__init__()
        
        self.img_size = img_size
        positional_embedding = positional_embedding if \
            positional_embedding in ['sine', 'learnable', 'none'] else 'sine'
        dim_feedforward = int(embedding_dim * mlp_ratio)
        self.embedding_dim = embedding_dim
        self.sequence_length = sequence_length
        self.seq_pool = seq_pool
        
        self.attention_pool = nn.Linear(self.embedding_dim, 256)
        self.positional_emb = nn.Parameter(
                    torch.zeros(1, sequence_length, embedding_dim),
                                                   requires_grad=True)
        # 词的长度后接embedding维度
        nn.init.trunc_normal_(self.positional_emb, std=0.2)
        
        self.dropout = nn.Dropout(p=dropout_rate)
        dpr = [x.item() for x in torch.linspace(
            0, stochastic_depth_rate, num_layers)] # 0-0.1区间等分12份
        self.blocks = nn.ModuleList([
            TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads,
                                    dim_feedforward=dim_feedforward,
                                    dropout=dropout_rate,
                                    attention_dropout=attention_dropout, 
                                    drop_path_rate=dpr[i])
            for i in range(num_layers)])
        # 把12个TransformerEncoderLayer编入subModule中,当作一个block
        self.norm = nn.LayerNorm(embedding_dim)
        
        self.conv_final = Conv2dFinal(1,1,3)
        
        self.apply(self.init_weight)
        
        
    def forward(self,x):
        if self.positional_emb is None and x.size(1) < self.sequence_length:
            x = F.pad(x, (0, 0, 0, self.n_channels - x.size(1)), 
                      mode='constant', value=0)

        if not self.seq_pool:
            cls_token = self.class_emb.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)
            
        # 这里要加上clone以免出现反向传播时的错误
        x = x.clone().reshape(-1,self.sequence_length,self.sequence_length)
        if self.positional_emb is not None:
            x += self.positional_emb  # 是值的直接加,[128,256,256]+[1,256,256]

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
        
        x = x.unsqueeze(1)
        x = self.conv_final(x)
        return x
    
    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.1)
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
    
    
class CXNOT(nn.Module):
    def __init__(self,img_size,
                 embedding_dim=256,
                 n_input_channels=3,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 pooling_kernel_size=3,
                 pooling_stride=2,
                 pooling_padding=1,
                 *args, **kwargs):
        super(CXNOT, self).__init__()
        
        self.tokenizer = OnlyTransformerToken(sequence_length=256,
                                            embedding_dim=embedding_dim,
                                            img_size = img_size,
                                            seq_pool=True,
                                            features=None,
                                            dropout=0.,
                                            attention_dropout=0.1,
                                            stochastic_depth=0.1,
                                            *args, **kwargs)
        self.conv_first = Conv2dReLUNoPooling(3,1,3)    # 先压成一个
        self.conv_1_4 = Conv2dReLU(1, 4, 3) # 128
        self.conv_4_16 = Conv2dReLU(4, 16, 3) # 64
        self.conv_16_64 = Conv2dReLU(16, 64, 3) # 32
        self.conv_64_256 = Conv2dReLU(64, 256, 3) # 16
        self.conv_256_1024 = Conv2dReLU(256, 1024, 3) # 8
        
        self.upsampling = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv_up_1024_256 = Conv2dReLUNoPooling(1024,256,3)
        self.conv_up_512_64 = Conv2dReLUNoPooling(512, 64, 3)
        self.conv_up_128_16 = Conv2dReLUNoPooling(128, 16, 3)
        self.conv_up_32_4 = Conv2dReLUNoPooling(32, 4, 3)
        self.conv_up_8_1 = Conv2dReLUNoPooling(8, 1, 3)
        self.conv_final = Conv2dFinal(2, 1, 3)

    def forward(self,x):
        x = self.conv_first(x) # 一开始的3变1
        attach_256 = self.tokenizer(x)
        x = self.conv_1_4(x) # 128,128
        attach_128 = self.tokenizer(x.reshape(-1,1,256,256)).reshape(-1,4,128,128)
        x = self.conv_4_16(x) # 64,64
        attach_64 = self.tokenizer(x.reshape(-1,1,256,256)).reshape(-1,16,64,64)
        x = self.conv_16_64(x) # 32,32
        attach_32 = self.tokenizer(x.reshape(-1,1,256,256)).reshape(-1,64,32,32)
        x = self.conv_64_256(x) # 16,16
        attach_16 = self.tokenizer(x.reshape(-1,1,256,256)).reshape(-1,256,16,16)
        x = self.conv_256_1024(x) # 8,8
        x = self.tokenizer(x.reshape(-1,1,256,256)).reshape(-1,1024,8,8)
        x = self.upsampling(x) # 16
        x = self.conv_up_1024_256(x)
        x = torch.cat((attach_16, x), dim=1)
        x = self.conv_up_512_64(x)  # 64,16,16
        x = self.upsampling(x)  # (64,32,32)
        x = torch.cat((attach_32, x), dim=1)
        x = self.conv_up_128_16(x) # (16,32,32)
        x = self.upsampling(x)  # (16,64,64)
        x = torch.cat((attach_64, x), dim=1)
        x = self.conv_up_32_4(x)
        x = self.upsampling(x)  # (4,128,128)
        x = torch.cat((attach_128, x), dim=1)
        x = self.conv_up_8_1(x)
        x = self.upsampling(x)  # (1,256,256)
        x = torch.cat((attach_256, x), dim=1)
        x = self.conv_final(x)
        
        return x
    

def _cxnot(num_layers, num_heads, mlp_ratio, embedding_dim,
            img_size,
         kernel_size=3, stride=None, padding=None,
         *args, **kwargs):
    return CXNOT(
                img_size = img_size,
                num_layers=num_layers,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                embedding_dim=embedding_dim,
                kernel_size=kernel_size,
                *args, **kwargs)
  
        
        
class CxnTokenizer(nn.Module):
    def __init__(self,
                 img_size,
                 stride=2,
                 kernel_size=3,
                 padding=1,
                 pooling_kernel_size=3,
                 pooling_stride=2, 
                 pooling_padding=1,
                 activation=None,
                 max_pool=True,
                 use_batchnorm=True):
        super(CxnTokenizer, self).__init__()
        
        self.img_size = img_size
        # Conv2dReLU默认stride=2
        self.conv1 = Conv2dReLU(3,4,kernel_size=3,padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv2dReLU(8,16,kernel_size=3,padding=1,
            use_batchnorm=use_batchnorm,
        )  # 把一个conv block集成了起来,加上了MaxPool2d,2倍的降采样
        self.conv3 = Conv2dReLU(32,64,kernel_size=3,padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv4 = Conv2dReLU(128,256,kernel_size=3,padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv5 = Conv2dReLUNoPooling(512,256,kernel_size=3,padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.flattener = nn.Flatten(2, 3) # 把第2维和第3维压平
        self.apply(self.init_weight)  # apply用于初始化

    def sequence_length(self, n_channels=3, height=256, width=256):
        a,_ = self.forward(torch.zeros((1, n_channels, height, width)))
        return a.shape[1]

    def forward(self, x):
        '''
        把经过conv_layers的shape类似于[1,3,224,224]的图像展平
        用在我的模型中,这里也要改,我的模型为[1,3,512,512],这里要变为[256,16,16]
        '''
        output_feature=[]
        b = x[:,0,:,:].unsqueeze(1)
        b0 = b.reshape([-1,4,int(self.img_size/2),int(self.img_size/2)])
        b1 = b.reshape([-1,16,int(self.img_size/4),int(self.img_size/4)])
        b2 = b.reshape([-1,64,int(self.img_size/8),int(self.img_size/8)])
        b3 = b.reshape([-1,256,int(self.img_size/16),int(self.img_size/16)])
        output_feature.append(b0)    # 保留初始的feature,就是wrap图
        output_feature.append(b1)
        output_feature.append(b2)
        output_feature.append(b3)
        
        x = self.conv1(x)  # [n,4,256,256]
        output_feature.append(x)
        x = torch.cat((output_feature[0], x), dim=1)
        
        x = self.conv2(x)  # [n,64,128,128]
        output_feature.append(x)
        x = torch.cat((output_feature[1], x), dim=1)
        
        x = self.conv3(x)  # [n,128,64,64]
        output_feature.append(x)
        x = torch.cat((output_feature[2], x), dim=1)
        
        x = self.conv4(x)  # [n,256,16,16] 保证这个与b一致
        output_feature.append(x)
        x = torch.cat((output_feature[3], x), dim=1)       
        x = self.conv5(x)
        
        # 可以return两个值
        output = self.flattener(x.transpose(-2, -1))
        return output,output_feature

    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight,mean=0,std=0.5)



class CXNTransformerUnetWithNoOrigin(nn.Module):
    '''
    这个是不带入origin data的
    '''
    def __init__(self, features,add_all_features,
                 seq_pool=True,
                 embedding_dim=256,
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
        self.add_all_features = add_all_features # 是否feature加到3层

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
                # 词的长度后接embedding维度
                nn.init.trunc_normal_(self.positional_emb, std=0.2)
                # 用从截断正态分布中提取的值填充输入张量
                # embedding_dim决定了positional_emb的维度,他要和x一样
            else:
                self.positional_emb = nn.Parameter(
                    self.sinusoidal_embedding(sequence_length, embedding_dim),
                                                   requires_grad=False) # sin
        else:
            self.positional_emb = None

        self.dropout = nn.Dropout(p=dropout_rate)
        dpr = [x.item() for x in torch.linspace(
            0, stochastic_depth_rate, num_layers)] # 0-0.1区间等分12份
        self.blocks = nn.ModuleList([
            TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads,
                                    dim_feedforward=dim_feedforward,
                                    dropout=dropout_rate,
                                    attention_dropout=attention_dropout, 
                                    drop_path_rate=dpr[i])
            for i in range(num_layers)])
        # 把12个TransformerEncoderLayer编入subModule中,当作一个block
        self.norm = nn.LayerNorm(embedding_dim)

        '''
        注意,改动以下部分让其输出为256x256维度的结果
        '''
        if not add_all_features:
            self.upsampling = nn.UpsamplingBilinear2d(scale_factor=2)
            self.conv_up_768_64 = Conv2dReLUNoPooling(512,64,kernel_size=3,padding=1,
                use_batchnorm=False,)
            self.conv_up_192_16 = Conv2dReLUNoPooling(128,16,kernel_size=3,padding=1,
                use_batchnorm=False,)
            self.conv_up_48_4 = Conv2dReLUNoPooling(32,4,kernel_size=3,padding=1,
                use_batchnorm=False,)
            self.conv_up_12_1 = Conv2dReLUNoPooling(8,4,kernel_size=3,padding=1,
                use_batchnorm=False,)
            self.final_output_0 = Conv2dFinal(4,4,kernel_size=7,padding=3,)
            self.final_output_1 = Conv2dFinal(4,1,kernel_size=3,padding=1,)
        else:
            self.upsampling = nn.UpsamplingBilinear2d(scale_factor=2)
            self.conv_up_768_64 = Conv2dReLUNoPooling(768,64,kernel_size=3,padding=1,
                use_batchnorm=False,)
            self.conv_up_192_16 = Conv2dReLUNoPooling(192,16,kernel_size=3,padding=1,
                use_batchnorm=False,)
            self.conv_up_48_4 = Conv2dReLUNoPooling(48,4,kernel_size=3,padding=1,
                use_batchnorm=False,)
            self.conv_up_12_1 = Conv2dReLUNoPooling(12,8,kernel_size=3,padding=1,
                use_batchnorm=False,)
            self.final_output_0 = Conv2dFinal(8,8,kernel_size=7,padding=3,)
            self.final_output_1 = Conv2dFinal(8,1,kernel_size=3,padding=1,)
        
        # self.fc1 = nn.Linear(256*256*32,12)  # 之前的模型遇到错误就把这个搞回来
        # self.fc2 = nn.Linear(12,1)
        self.apply(self.init_weight)


    def forward(self, x,features):
        if self.positional_emb is None and x.size(1) < self.sequence_length:
            x = F.pad(x, (0, 0, 0, self.n_channels - x.size(1)), 
                      mode='constant', value=0)

        if not self.seq_pool:
            cls_token = self.class_emb.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)

        if self.positional_emb is not None:
            x += self.positional_emb  # 是值的直接加,[128,256,256]+[1,256,256]

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
        
        x = x.reshape([-1,256,16,16])   # 这一步时max和min都对
        if self.add_all_features:
            x = torch.cat((features[3], x), dim=1)
        x = torch.cat((features[-1], x), dim=1)
        x = self.upsampling(x) # [64,64]
        x = self.conv_up_768_64(x)
        
        if self.add_all_features:
            x = torch.cat((features[2], x), dim=1)  
        x = torch.cat((features[-2], x), dim=1)
        x = self.upsampling(x) # [128,128]
        x = self.conv_up_192_16(x)
        
        if self.add_all_features:
            x = torch.cat((features[1], x), dim=1) # 把原图拼接进去
        x = torch.cat((features[-3], x), dim=1)
        x = self.upsampling(x) # [256,256]
        x = self.conv_up_48_4(x)
        
        if self.add_all_features:
            x = torch.cat((features[0], x), dim=1) # [n,64,256,256]
        x = torch.cat((features[4], x), dim=1)
        x = self.upsampling(x)
        x = self.conv_up_12_1(x)
        x = self.final_output_0(x)
        x = self.final_output_1(x)
        # xx = self.fc1(x.flatten())
        # xx = self.fc2(xx)
        return x

    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.2)
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
    def __init__(self,img_size,add_all_features,
                 embedding_dim=256,
                 n_input_channels=3,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 pooling_kernel_size=3,
                 pooling_stride=2,
                 pooling_padding=1,
                 *args, **kwargs):
        super(CXNCCT, self).__init__()
        '''
        Tokenizer得输出3步降采样的features
        '''
        self.tokenizer = CxnTokenizer(img_size=img_size,
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   padding=padding,
                                   pooling_kernel_size=pooling_kernel_size,
                                   pooling_stride=pooling_stride,
                                   pooling_padding=pooling_padding,
                                   max_pool=True,
                                   activation=nn.ReLU)
        
        self.wrapUnet = CXNTransformerUnetWithNoOrigin(
            add_all_features = add_all_features,
            sequence_length=self.tokenizer.sequence_length(n_channels=n_input_channels,
                                                           height=img_size,
                                                           width=img_size),
            embedding_dim=embedding_dim,
            seq_pool=True,
            features=None,
            dropout=0.,
            attention_dropout=0.1,
            stochastic_depth=0.1,
            *args, **kwargs)    # CXNTransformerUnetWithNoOrigin二选一

    def forward(self, x):   # 这里传的features
        x,features = self.tokenizer(x)
        return self.wrapUnet(x,features)    


def _cxncct(num_layers, num_heads, mlp_ratio, embedding_dim,
            img_size,add_all_features,
         kernel_size=3, stride=None, padding=None,
         *args, **kwargs):
    stride = stride if stride is not None else max(1, (kernel_size // 2) - 1)
    padding = padding if padding is not None else max(1, (kernel_size // 2))
    return CXNCCT(
                img_size = img_size,
                add_all_features = add_all_features,
                num_layers=num_layers,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                embedding_dim=embedding_dim,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                *args, **kwargs)


def cct_2(*args, **kwargs):   # 这里注意embedding_dim
    return _cxncct(num_layers=2, num_heads=2, mlp_ratio=1, embedding_dim=256,
                *args, **kwargs)


def cct_4(*args, **kwargs):
    return _cxncct(num_layers=4, num_heads=2, mlp_ratio=1, embedding_dim=256,
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
    return _cxncct(num_layers=10, num_heads=8, mlp_ratio=3, embedding_dim=256,
                *args, **kwargs)


def cct_12(*args, **kwargs):
    return _cxncct(num_layers=12, num_heads=12, mlp_ratio=4, embedding_dim=768,
                *args, **kwargs)


def cot_2(*args, **kwargs):   # 这里注意embedding_dim
    return _cxnot(num_layers=2, num_heads=2, mlp_ratio=1, embedding_dim=256,
                *args, **kwargs)


def cot_4(*args, **kwargs):
    return _cxnot(num_layers=4, num_heads=2, mlp_ratio=1, embedding_dim=256,
                *args, **kwargs)


def cot_6(*args, **kwargs):
    return _cxnot(num_layers=6, num_heads=4, mlp_ratio=2, embedding_dim=256,
                *args, **kwargs)


def cot_7(*args, **kwargs):
    return _cxnot(num_layers=7, num_heads=4, mlp_ratio=2, embedding_dim=256,
                *args, **kwargs)


def cot_8(*args, **kwargs):
    return _cxnot(num_layers=8, num_heads=4, mlp_ratio=2, embedding_dim=256,
                *args, **kwargs)


def cot_10(*args, **kwargs):
    return _cxnot(num_layers=10, num_heads=8, mlp_ratio=3, embedding_dim=256,
                *args, **kwargs)

