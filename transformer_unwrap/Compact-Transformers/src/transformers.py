# Thanks to rwightman's timm package
# github.com:rwightman/pytorch-image-models


import torch
from torch.nn import Module, Linear, Dropout, LayerNorm, Identity
import torch.nn.functional as F


class Attention(Module):
    """
    Obtained from: github.com:rwightman/pytorch-image-models
    attention解释:
    https://segmentfault.com/a/1190000017899526?utm_source=tag-newest
    """
    def __init__(self, dim, num_heads=8, attention_dropout=0.1,
                 projection_dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        '''
        此处的dim是embedding_dim,Embedding 是一个将离散变量转为连续向量表示的
        一个方式,dim是总维度,维度过低表示能力不够，维度过高容易过拟合
        '''
        head_dim = dim // self.num_heads  # attention每个head有几个维度
        self.scale = head_dim ** -0.5    # 缩放比例是head_dim开根号

        self.qkv = Linear(dim, dim * 3, bias=False)
        '''
        首先分别对V，K，Q三者分别进行线性变换,即将三者分别输入到三个单层神经网络层,
        激活函数选择relu,输出新的V,K,Q(三者shape都和原来shape相同,
                             即经过线性变换时输出维度和输入维度相同
        query, key, value
        '''
        self.attn_drop = Dropout(attention_dropout)
        self.proj = Linear(dim, dim)
        self.proj_drop = Dropout(projection_dropout)

    def forward(self, x):   
        # x就是输入的张量
        B, N, C = x.shape
        '''
        然后将Q在最后一维上进行切分为num_heads,假设为8,然后对切分完的矩阵
        在axis=0维上进行concat链接起来,对V和K都进行和Q一样的操作;
        操作后的矩阵记为Q_,K_,V_. 例如从[1,10,512]变为[8,10,64]
        '''
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, 
                                  C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        '''
        将Q_,K_.T相乘并Scale,得到的output,得到的output为[8,10,10],
        执行output = softmax(output),
        '''
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        '''
        然后将更新后的output想乘V_,
        得到再次更新后的output矩阵[8,10,64]，然后将得到的output在0维上切分为8段;
        在2维上合并为[10，512]原始shape样式
        '''
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        # transpose（0，2，1）：表示Y轴与Z轴发生轴变换
        x = self.proj(x)  # 最后再进行一个linear层变换
        x = self.proj_drop(x)
        return x


class TransformerEncoderLayer(Module):
    """
    Inspired by torch.nn.TransformerEncoderLayer and
    rwightman's timm package.
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 attention_dropout=0.1, drop_path_rate=0.1):
        '''
        d_model: embedding dim(cct中默认是768); nhead=num_heads(cct中默认是12)
        
        '''
        super(TransformerEncoderLayer, self).__init__()
        self.pre_norm = LayerNorm(d_model)
        '''
        LayerNorm：channel方向做归一化，算CHW的均值，主要对RNN作用明显；
        d_model表示normalized shape
        numpy实现pytorch无参数版本layernorm:
        mean = np.mean(a.numpy(), axis=(1,2))
        var = np.var(a.numpy(), axis=(1,2))
        div = np.sqrt(var+1e-05)
        ln_out = (a-mean[:,None,None])/div[:,None,None]
        '''
        self.self_attn = Attention(dim=d_model, num_heads=nhead,
                                   attention_dropout=attention_dropout, 
                                   projection_dropout=dropout)

        self.linear1 = Linear(d_model, dim_feedforward) # (768,2048)
        self.dropout1 = Dropout(dropout)
        self.norm1 = LayerNorm(d_model)
        self.linear2 = Linear(dim_feedforward, d_model) # (2048,768)
        self.dropout2 = Dropout(dropout)
        # identity模块不改变输入,直接return input
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else Identity()

        self.activation = F.gelu

    def forward(self, src: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        src = src + self.drop_path(self.self_attn(self.pre_norm(src)))
        src = self.norm1(src)
        src2 = self.linear2(self.dropout1(self.activation(self.linear1(src))))
        src = src + self.drop_path(self.dropout2(src2))
        return src


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Obtained from: github.com:rwightman/pytorch-image-models
    Drop paths (Stochastic Depth) per sample (when applied in main path 
                                              of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, 
    etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form 
    of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than 
    mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    '''
    函数参数中的冒号是参数的类型建议符，告诉程序员希望传入的实参的类型。
    函数后面跟着的箭头是函数返回值的类型建议符，用来说明该函数返回的值是什么类型
    '''
    if drop_prob == 0. or not training: # 如果不drop，则返回原值
        return x
    keep_prob = 1 - drop_prob
    # x.ndim表示是几维张量
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # [shape[0],1,1]
    # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, 
                                           dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(Module):
    """
    Obtained from: github.com:rwightman/pytorch-image-models
    Drop paths (Stochastic Depth) per sample  
    (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
