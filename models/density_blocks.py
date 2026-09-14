"""复用Pocket_Plus的残差三维卷积与解码门控；不包含循环状态或任务输出头。"""
import torch
import einops
from torch import nn
from torch.utils.checkpoint import checkpoint as torch_checkpoint

class Bottleneck(nn.Module):
    """
    使用1x1卷积, 3x3卷积, 1x1卷积. 
    
    输入参数 (Input Parameters):
        - in_planes: int, 输入特征图的通道数
        - planes: int, 中间层的基础通道数(瓶颈处通道数)
        - stride: int, 默认=1, 卷积步长, stride=2时进行下采样
        - groups: int, 默认=1, 分组卷积的组数
        - activation_class: nn.Module, 默认=nn.ReLU, 激活函数类
        - conv_class: nn.Module, 默认=nn.Conv3d, 卷积层类
        - affine: bool, 默认=False, InstanceNorm3d是否使用可学习的仿射参数
        - checkpoint: bool, 默认=False, 是否使用梯度检查点(节省显存)
        - **kwargs: 其他关键字参数
    
    类属性 (Class Attributes):
        - expansion: int = 4, 通道扩展倍数, 输出通道数=planes * expansion
    
    输出 (Output):
        - forward返回: torch, (B, planes*expansion, D', H', W'), 其中D', H', W'由stride决定, stride=2时各维度减半
    """
    expansion = 4

    def __init__(
        self,
        in_planes,
        planes,
        stride=1,
        groups=1,
        activation_class=nn.ReLU,
        conv_class=nn.Conv3d,
        affine=False,
        checkpoint=False,
        **kwargs,
    ):
        super().__init__()
        self.activation_fn = activation_class()
        self.conv1 = conv_class(
            in_planes, planes, kernel_size=1, bias=False, groups=groups
        )
        self.norm1 = nn.InstanceNorm3d(planes, affine=affine)
        self.conv2 = conv_class(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
            groups=groups,
        )
        self.norm2 = nn.InstanceNorm3d(planes, affine=affine)
        self.conv3 = conv_class(
            planes, self.expansion * planes, kernel_size=1, bias=False, groups=groups
        )
        self.norm3 = nn.InstanceNorm3d(self.expansion * planes, affine=affine)

        # 当stride!=1或通道数不匹配时, 需要用1x1卷积调整维度
        self.shortcut_conv = nn.Identity()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut_conv = nn.Conv3d(
                in_planes,
                self.expansion * planes,
                kernel_size=1,
                stride=stride,
                bias=False,
                groups=groups,
            )
        # 根据checkpoint参数选择是否使用梯度检查点
        self.forward = self.forward_checkpoint if checkpoint else self.forward_normal


    def forward_normal(self, x):
        """
        普通前向传播
        """
        out = self.activation_fn(self.norm1(self.conv1(x)))
        out = self.activation_fn(self.norm2(self.conv2(out)))
        out = self.norm3(self.conv3(out))
        out += self.shortcut_conv(x)
        out = self.activation_fn(out)
        return out

    def forward_checkpoint(self, x):
        """
        带梯度检查点的前向传播(节省显存)
        """
        return torch_checkpoint(self.forward_normal, x, preserve_rng_state=False)





class ConvBuildingBlock(nn.Module):
    """
    双层3x3卷积+最后残差, 空间维度不变
    """
    def __init__(self, in_channels:int, out_channels:int, activate_class:nn.Module=nn.ReLU):
        super().__init__()
        self.activate_function = activate_class()
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels,affine=True),
            self.activate_function,
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
        )
        
        self.shortcut_conv = nn.Identity()
        if in_channels != out_channels:
            self.shortcut_conv = nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                bias=True
            )
    
    def forward(self,x:torch.Tensor):
        return self.activate_function(self.conv1(x) + self.shortcut_conv(x))




class ShortConv(nn.Module):
    """
    简单3x3卷积块, 空间维度不变
    """
    def __init__(self,in_channels:int,out_channels:int):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm3d(out_channels,affine=True)
        self.relu1 = nn.ReLU()
    
    def forward(self,x:torch.Tensor):
        y = self.norm1(self.conv1(x))
        y = self.relu1(y)
        return y




class ShortConvAdd(nn.Module):
    """
    2个特征 x_0, x_1 的融合. x_0 为原始输入特征(如13通道的输入), x_1 为循环传递的特征(如64通道的recycle特征)
    
    输入参数 (Input Parameters):
        - input_channels: int 或 None, 第一个输入(x0)的通道数; None 时使用 LazyConv3d 延迟初始化
        - output_channels: int, 输出通道数, 同时也是第二个输入(x1)的通道数
    """
    def __init__(self, input_channels, output_channels: int):
        super().__init__()
        self.output_channels = output_channels
        if input_channels is None:
            # 延迟初始化: 在第一次 forward 时根据输入自动推断 in_channels
            self.conv1 = nn.LazyConv3d(output_channels, kernel_size=3, stride=1, padding=1, bias=False)
        else:
            self.conv1 = nn.Conv3d(input_channels, output_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm3d(output_channels, affine=False)
        self.norm2 = nn.InstanceNorm3d(output_channels, affine=True)
        self.relu1 = nn.ELU()
    
    def forward(self,x0,x1):
        y = self.norm1(self.conv1(x0))
        y = self.relu1( y + self.norm2(x1))
        return y





class Res2NetBlock(nn.Module):
    """
    层级残差模块
    
    输入参数 (Input Parameters):
        - scale: int, 默认=4, 分割尺度数 or 层级数
    """
    def __init__(self, in_channels, out_channels, stride=1, scale=4,activate_class:nn.Module=nn.ReLU):
        super(Res2NetBlock, self).__init__()
        self.scale = scale
        self.conv1 = nn.Sequential(nn.Conv3d(in_channels, out_channels*self.scale, kernel_size=1, stride=1, padding=0, bias=False), nn.InstanceNorm3d(out_channels*self.scale, affine=True))
        self.norm1 = nn.InstanceNorm3d(out_channels*self.scale, affine=True)
        self.conv_list = nn.ModuleList([nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False) for _ in range(self.scale - 1)])
        self.activate_class = activate_class()
        self.conv2 = nn.Sequential(nn.Conv3d(out_channels*self.scale, out_channels, 1, 1, 0, bias=False), nn.InstanceNorm3d(out_channels, affine=True))
        
        self.shortcut_conv = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut_conv = nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=True
            )
    
    def forward(self, x):
        # x_list: tuple of torch tensors, 每个元素形状为(B, out_channels, D, H, W)
        x_list = self.activate_class(self.conv1(x)).chunk(self.scale,dim=1)  # 将扩展后的特征在通道维度上分割成scale份
        # y_list: list of torch tensors, 存储级联卷积的输出
        y_list = []
        for ii,xi in enumerate(x_list):
            if ii == 0:
                y_list.append(xi)
            elif ii == 1:
                y_list.append(self.conv_list[ii-1](xi))
            else:
                y_list.append(self.conv_list[ii-1](xi+y_list[-1]))
        
        y = self.conv2(self.activate_class(self.norm1(  torch.cat(y_list,dim=1)  )))
        y = self.activate_class(y+self.shortcut_conv(x))
        return y
    




class AttentionGate(nn.Module):
    """
    输入参数 (Input Parameters):
        - down_features: int, 下采样路径(跳跃连接)特征的通道数
        - up_features: int, 上采样路径特征的通道数
        - out_features: int, 输出特征的通道数
        - attention_features: int, 默认=64, 注意力计算中间特征的通道数
        - attention_heads: int, 默认=8, 注意力头的数量————规定 up_features = afz * ahz
    
    输出 (Output):
        - forward返回: torch, (B, out_features, D, H, W), 融合后的特征图
    """
    def __init__(self, down_features:int, up_features:int, out_features:int, attention_features:int=64, attention_heads:int=8):
        super(AttentionGate, self).__init__()
        self.dfz = down_features
        self.ufz = up_features
        self.ofz = out_features
        self.afz = attention_features
        self.ahz = attention_heads
        
        # (up_features) -> (attention_features)
        self.conv_q = nn.Sequential(nn.Conv3d(
            in_channels=self.ufz,
            out_channels=self.afz,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        ),nn.InstanceNorm3d(self.afz,affine=True))
        
        # (down_features) -> (attention_features)
        self.conv_k = nn.Sequential(nn.Conv3d(
            in_channels=self.dfz,
            out_channels=self.afz,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        ),nn.InstanceNorm3d(self.afz,affine=True))
        
        # (down_features) -> (up_features), 规定 up_features = afz * ahz
        self.conv_v = nn.Sequential(nn.Conv3d(
            in_channels=self.dfz,
            out_channels=self.ufz,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        ),nn.InstanceNorm3d(self.ufz,affine=True))
        
        # (attention_features) -> (attention_heads), 输出经Sigmoid归一化到[0,1]
        self.gate = nn.Sequential(
            nn.ReLU(),
            nn.Conv3d(
                in_channels=self.afz,
                out_channels=self.ahz,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True
            ),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU()
        # (up_features) -> (out_features)
        self.conv_back = ConvBuildingBlock(self.ufz, self.ofz)
    
    def forward(self,us,ds):
        """
        输入参数 (Input Parameters):
            - us: torch, (B, up_features, D_us, H_us, W_us), 上采样路径的特征(较粗分辨率)
            - ds: torch, (B, down_features, D, H, W), 跳跃连接的特征(较细分辨率)
        
        输出 (Output):
            - torch, (B, out_features, D, H, W), 注意力加权融合后的特征
        """
        ds_shape = ds.shape
        D, H, W = ds_shape[2:]
        # 将上采样特征插值到与跳跃连接相同的分辨率
        upsampled = nn.functional.interpolate(input=us, size=(D, H, W), mode='trilinear', align_corners=True)
        query = self.conv_q(upsampled)
        key = self.conv_k(ds)
        value = self.conv_v(ds)
        
        # value: torch, (B, afz, ahz, D, H, W)
        # 将value重排为多头形式, 其中 up_features = afz * ahz
        value = einops.rearrange(value, "N (afz ahz) d h w -> N afz ahz d h w", ahz=self.ahz)
        
        # gate: torch, (B, ahz, D, H, W), 注意力权重, 范围[0,1]
        # 通过query和key的加法融合计算得到
        gate = self.gate(query+key)  # N ahz d h w
        
        # out: torch, (B, afz, ahz, D, H, W)
        # 对value应用注意力权重(广播乘法)
        out = value*gate[:,None]
        
        # out: torch, (B, up_features, D, H, W)
        # 将多头输出重排回原始形式
        out = einops.rearrange(out,"N afz ahz d h w -> N (afz ahz) d h w", ahz=self.ahz)
        
        # 将注意力加权的输出与上采样特征相加, 经ReLU激活后通过输出卷积块
        return self.conv_back(self.relu(out+upsampled))
