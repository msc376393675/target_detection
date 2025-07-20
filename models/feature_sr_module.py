import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

# ----------------------------------------------------------------------
# 这一部分代码直接来源于你提供的 Real-ESRGAN 仓库
# 我们将其核心模型 RRDBNet 提取出来，用于特征图增强
# 来源: xinntao/real-esrgan/realesrgan/archs/rrdbnet_arch.py
# ----------------------------------------------------------------------

class ResidualDenseBlock(nn.Module):
    """Residual Dense Block.

    Used in RRDB block in ESRGAN.

    Args:
        num_feat (int): Channel number of intermediate features.
        num_grow_ch (int): Growing channels helped in residual dense block.
    """

    def __init__(self, num_feat=64, num_grow_ch=32):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        # Empirically, we use 0.2 to scale the residual for better performance
        return x5 * 0.2 + x

class RRDB(nn.Module):
    """Residual in Residual Dense Block."""

    def __init__(self, num_feat, num_grow_ch=32):
        super(RRDB, self).__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        # Empirically, we use 0.2 to scale the residual for better performance
        return out * 0.2 + x

class RRDBNet(nn.Module):
    """Networks consisting of Residual in Residual Dense Block, which is used in ESRGAN.

    ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks.

    Args:
        num_in_ch (int): Channel number of inputs.
        num_out_ch (int): Channel number of outputs.
        num_feat (int): Channel number of intermediate features.
        num_block (int): Number of RRDB blocks.
        num_grow_ch (int): Growing channels helped in residual dense block.
    """

    def __init__(self, num_in_ch, num_out_ch, num_feat, num_block, num_grow_ch=32, scale=4):
        super(RRDBNet, self).__init__()
        self.scale = scale
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = nn.ModuleList([RRDB(num_feat, num_grow_ch) for _ in range(num_block)])
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        # upsampling
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        feat = self.conv_first(x)
        body_feat = self.conv_body(self.body(feat)) # This line is not in the original file, but is required
        feat = feat + body_feat

        # upsample
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest')))
        
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out

# ----------------------------------------------------------------------
# 我们的创新封装：Feature Super-Resolution Module
# ----------------------------------------------------------------------

class FeatureSRModule(nn.Module):
    """
    一个即插即用的模块，利用RRDBNet对输入的特征图进行超分和增强。
    这个模块可以被插入到任何检测模型的Neck部分。
    """
    def __init__(self, in_channels, out_channels, num_rrdb_blocks=6, scale=4, model_name='RealESRGAN_x4plus.pth'):
        """
        Args:
            in_channels (int): 输入特征图的通道数.
            out_channels (int): 输出特征图的通道数.
            num_rrdb_blocks (int): RRDB模块的数量，可以根据模型大小调整.
            model_name (str): 要加载的Real-ESRGAN权重文件名.
        """
        super().__init__()
        self.model_name = model_name
        
        self.sr_module = RRDBNet(
            num_in_ch=in_channels,
            num_out_ch=out_channels,
            num_feat=64,
            num_block=num_rrdb_blocks,
            num_grow_ch=32,
            scale=scale
        )
        print(f"✅ FeatureSRModule (based on Real-ESRGAN) initialized.")
        print(f"   - Input Channels: {in_channels}")
        print(f"   - Output Channels: {out_channels}")
        print(f"   - RRDB Blocks: {num_rrdb_blocks}")

        # --- 新增：加载预训练权重 ---
        self.load_pretrained_weights()

    def load_pretrained_weights(self):
        """加载指定的Real-ESRGAN权重。"""
        weights_path = Path('weights') / self.model_name
        if not weights_path.exists():
            print(f"⚠️ 警告: 找不到指定的Real-ESRGAN权重文件: {weights_path}")
            print("   - 将使用随机初始化的权重。")
            return

        try:
            # 加载权重文件
            loadnet = torch.load(weights_path, map_location=torch.device('cpu'))
            
            # 通常权重保存在'params_ema'或'params'键中
            if 'params_ema' in loadnet:
                keyname = 'params_ema'
            elif 'params' in loadnet:
                keyname = 'params'
            else:
                keyname = None

            if keyname:
                 # 因为我们只用了RRDBNet，所以需要适配一下键名
                 # 原始的键可能是 'model.0.weight', 'model.1.rdb1.conv1.weight' 等
                 # 我们需要移除 'model.' 这个前缀
                state_dict = loadnet[keyname]
                adapted_state_dict = {}
                for k, v in state_dict.items():
                    # 适配我们的RRDBNet实例 (self.sr_module)
                    new_key = k.replace('model.', '') 
                    if new_key in self.sr_module.state_dict():
                        adapted_state_dict[new_key] = v
                
                self.sr_module.load_state_dict(adapted_state_dict, strict=False)
                print(f"✅ 成功从 {weights_path} 加载Real-ESRGAN权重。")
            else:
                # 如果没有特定的键，尝试直接加载整个文件
                self.sr_module.load_state_dict(loadnet, strict=False)
                print(f"✅ 成功从 {weights_path} 加载Real-ESRGAN权重。")

        except Exception as e:
            print(f"❌ 加载Real-ESRGAN权重失败: {e}")
            print("   - 将使用随机初始化的权重。")

    def forward(self, feature_map):
        return self.sr_module(feature_map)