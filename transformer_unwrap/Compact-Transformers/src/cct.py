import torch
import torch.nn as nn
import torch.nn.functional as F
from .transformers import TransformerEncoderLayer
# from transformers import TransformerEncoderLayer


__all__ = ['cot_2', 'cot_512','cot_1024', 'otu_256','otu_512','otu_1024']


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
        relu = nn.ReLU(inplace=True)
        
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
        relu = nn.ReLU(inplace=True)

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
            avg_pool_size=256,
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
        avgpool = nn.AdaptiveAvgPool2d(avg_pool_size)   # 最后加一个avgpool
        super(Conv2dFinal, self).__init__(conv,avgpool)
        
        
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
        
        self.attention_pool = nn.Linear(self.embedding_dim, self.sequence_length)
        self.positional_emb = nn.Parameter(
                    torch.zeros(1, self.embedding_dim, self.sequence_length),
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
        
        self.conv_final = Conv2dFinal(1,1,3,avg_pool_size=img_size)
        
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
        
        self.dim = embedding_dim
        self.tokenizer1 = OnlyTransformerToken(sequence_length=self.dim,
                                            embedding_dim=self.dim,
                                            img_size = img_size,
                                            seq_pool=True,
                                            features=None,
                                            dropout=0.,
                                            attention_dropout=0.1,
                                            stochastic_depth=0.1,
                                            *args, **kwargs)
        self.tokenizer2 = OnlyTransformerToken(sequence_length=self.dim,
                                            embedding_dim=self.dim,
                                            img_size = img_size,
                                            seq_pool=True,
                                            features=None,
                                            dropout=0.,
                                            attention_dropout=0.1,
                                            stochastic_depth=0.1,
                                            *args, **kwargs)
        self.tokenizer3 = OnlyTransformerToken(sequence_length=self.dim,
                                            embedding_dim=self.dim,
                                            img_size = img_size,
                                            seq_pool=True,
                                            features=None,
                                            dropout=0.,
                                            attention_dropout=0.1,
                                            stochastic_depth=0.1,
                                            *args, **kwargs)
        self.tokenizer4 = OnlyTransformerToken(sequence_length=self.dim,
                                            embedding_dim=self.dim,
                                            img_size = img_size,
                                            seq_pool=True,
                                            features=None,
                                            dropout=0.,
                                            attention_dropout=0.1,
                                            stochastic_depth=0.1,
                                            *args, **kwargs)
        self.tokenizer5 = OnlyTransformerToken(sequence_length=self.dim,
                                            embedding_dim=self.dim,
                                            img_size = img_size,
                                            seq_pool=True,
                                            features=None,
                                            dropout=0.,
                                            attention_dropout=0.1,
                                            stochastic_depth=0.1,
                                            *args, **kwargs)
        self.tokenizer6 = OnlyTransformerToken(sequence_length=self.dim,
                                            embedding_dim=self.dim,
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
        self.conv_final_0 = Conv2dReLUNoPooling(2, 4, 3)
        self.conv_final_1 = Conv2dFinal(4, 8, 3,avg_pool_size=self.dim)
        self.conv_final_2 = Conv2dFinal(8, 4, 3,avg_pool_size=self.dim)
        self.conv_final_3 = nn.Conv2d(4, 1, 3,padding=1,stride=1)

    def forward(self,x):
        x = self.conv_first(x) # 一开始的3变1
        attach_256 = self.tokenizer1(x)
        x = self.conv_1_4(x) # 128,128
        attach_128 = self.tokenizer2(x.reshape(
            -1,1,self.dim,self.dim)).reshape(-1,4,self.dim//2,self.dim//2)
        x = self.conv_4_16(x) # 64,64
        attach_64 = self.tokenizer3(x.reshape(
            -1,1,self.dim,self.dim)).reshape(-1,16,self.dim//4,self.dim//4)
        x = self.conv_16_64(x) # 32,32
        attach_32 = self.tokenizer4(x.reshape(
            -1,1,self.dim,self.dim)).reshape(-1,64,self.dim//8,self.dim//8)
        x = self.conv_64_256(x) # 16,16
        attach_16 = self.tokenizer5(x.reshape(
            -1,1,self.dim,self.dim)).reshape(-1,256,self.dim//16,self.dim//16)
        x = self.conv_256_1024(x) # 8,8
        x = self.tokenizer6(x.reshape(
            -1,1,self.dim,self.dim)).reshape(-1,1024,self.dim//32,self.dim//32)
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
        x = self.conv_final_0(x)
        x = self.conv_final_1(x)
        x = self.conv_final_2(x)
        x = self.conv_final_3(x)
        return x
    

def _cxnot(num_layers, num_heads, mlp_ratio, embedding_dim,img_size,
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
 
    
class OTUCXN(nn.Module):
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
        super(OTUCXN, self).__init__()
        
        self.dim = embedding_dim
        self.tokenizer_row = OnlyTransformerToken(sequence_length=self.dim,
                                            embedding_dim=self.dim,
                                            img_size = img_size,
                                            seq_pool=True,
                                            features=None,
                                            dropout=0.,
                                            attention_dropout=0.1,
                                            stochastic_depth=0.1,
                                            *args, **kwargs)
        
        self.tokenizer_col = OnlyTransformerToken(sequence_length=self.dim,
                                            embedding_dim=self.dim,
                                            img_size = img_size,
                                            seq_pool=True,
                                            features=None,
                                            dropout=0.,
                                            attention_dropout=0.1,
                                            stochastic_depth=0.1,
                                            *args, **kwargs)
        self.conv_first = Conv2dReLUNoPooling(3,1,3)    # 先压成一个
        self.conv_final_0 = Conv2dReLUNoPooling(2, 4, 3)
        self.conv_final_1 = Conv2dFinal(4, 8, 3,avg_pool_size=self.dim)
        self.conv_final_2 = Conv2dFinal(8, 4, 3,avg_pool_size=self.dim)
        self.conv_final_3 = nn.Conv2d(4, 1, 3,padding=1,stride=1)
        
    def forward(self,x):
        x = self.conv_first(x) # 一开始的3变1
        x = self.tokenizer_row(x)
        x = self.tokenizer_col(x)
        x = torch.cat((x, x), dim=1)
        x = self.conv_final_0(x)
        x = self.conv_final_1(x)
        x = self.conv_final_2(x)
        x = self.conv_final_3(x)
        return x


def _otucxn(num_layers, num_heads, mlp_ratio, embedding_dim,img_size,
         kernel_size=3, stride=None, padding=None,
         *args, **kwargs):
    return OTUCXN(
                img_size = img_size,
                num_layers=num_layers,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                embedding_dim=embedding_dim,
                kernel_size=kernel_size,
                *args, **kwargs)
        


def cot_2(*args, **kwargs):   # 这里注意embedding_dim
    return _cxnot(num_layers=2, num_heads=2, mlp_ratio=1, embedding_dim=256,
                *args, **kwargs)


def cot_512(*args, **kwargs):
    return _cxnot(num_layers=2, num_heads=2, mlp_ratio=1, embedding_dim=512,
                *args, **kwargs)


def cot_1024(*args, **kwargs):
    return _cxnot(num_layers=2, num_heads=2, mlp_ratio=1, embedding_dim=1024,
                *args, **kwargs)


def otu_256(*args, **kwargs):   # 这里注意embedding_dim
    return _otucxn(num_layers=2, num_heads=2, mlp_ratio=1, embedding_dim=256,
                *args, **kwargs)


def otu_512(*args, **kwargs):   # 这里注意embedding_dim
    return _otucxn(num_layers=2, num_heads=2, mlp_ratio=1, embedding_dim=512,
                *args, **kwargs)


def otu_1024(*args, **kwargs):   # 这里注意embedding_dim
    return _otucxn(num_layers=2, num_heads=2, mlp_ratio=1, embedding_dim=1024,
                *args, **kwargs)