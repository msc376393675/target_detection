# file: models/hidef_yolo.py

import torch
import torch.nn as nn
import torchvision.ops as ops

from .feature_sr_module import FeatureSRModule
from .transformer_head import TransformerRefinerHead

class HiDefYOLO(nn.Module):
    """
    Hi-Def-YOLO 论文的核心模型框架。
    结合了 YOLOv9 (Proposer), FeatureSRModule (Enhancer), 和 TransformerRefinerHead (Refiner).
    """
    def __init__(self, num_classes=80, sr_in_channels=256, feature_dim=256, yolov9_variant='yolov9-c.pt', sr_model_name='RealESRGAN_x4plus.pth'):
        super().__init__()

        # --- 第一阶段: 加载YOLOv9作为Proposer ---
        # 使用torch.hub来加载一个预训练的YOLOv9模型
        
        print(f"⌛ Loading YOLOv9 model ('{yolov9_variant}')...")
        self.yolo_proposer = torch.hub.load('ultralytics/yolov9', 'custom', path_or_model=yolov9_variant)
        print("✅ YOLOv9 Proposer loaded successfully.")
        
        # 不训练YOLOv9的骨干网络，只将其作为特征提取器和提议器
        for param in self.yolo_proposer.parameters():
            param.requires_grad = False
            
        # --- 第二阶段 (a): 初始化特征超分模块 ---
        # 这个模块将被用于增强从YOLOv9 Neck中提取的特征图
        self.feature_enhancer = FeatureSRModule(
            in_channels=sr_in_channels, 
            out_channels=feature_dim,
            num_rrdb_blocks=6,
            model_name=sr_model_name 
        )
        
        # --- 第二阶段 (b): 初始化Transformer精炼头 ---
        self.refiner_head = TransformerRefinerHead(
            feature_dim=feature_dim,
            num_heads=8,
            num_layers=2
        )
        
        self.num_classes = num_classes

    def forward(self, x, targets=None):
        """
        模型的前向传播逻辑。
        """
        # --- 第一阶段: 运行YOLOv9 Proposer ---
        # 从YOLOv9中获取两样东西：
        # 1. 初始的、粗略的检测框 (proposals)
        # 2. 来自Neck部分的特征图，用于后续的增强和精炼
        
        # YOLOv9的内部结构可能需要具体分析，这里用一个概念性的方式来表示
        # `yolo_proposer.model.model` 通常是其内部网络结构
        # 需要找到合适的中间层来提取特征
        
        # 假设yolo_proposer返回的是 (检测结果, [特征图列表])
        proposals, feature_maps = self.yolo_proposer.model.forward_once(x)
        
        initial_boxes = proposals[0][..., :4] # (x_center, y_center, w, h)
        initial_scores = proposals[0][..., 4]
        
        small_object_feature_map = feature_maps[-1] 
        
        
        # --- 第二阶段 (a): 运行特征增强模块 ---
        # 对选出的特征图进行超分和增强
        enhanced_feature_map = self.feature_enhancer(small_object_feature_map)
        
        
        # --- 第二阶段 (b): 准备并运行Transformer精炼头 ---
        
        # 1. 从初始检测框中提取对应的特征 (ROI Align)

        topk_scores, topk_indices = torch.topk(initial_scores[0], k=100)
        topk_boxes = initial_boxes[0][topk_indices]
        

        box_indices = torch.zeros(topk_boxes.size(0), dtype=torch.int, device=x.device)

        object_features = torch.rand(1, 100, self.refiner_head.feature_dim).to(x.device)
        # 将增强后的特征图作为Transformer的memory
        b, c, h, w = enhanced_feature_map.shape
        memory = enhanced_feature_map.flatten(2).permute(0, 2, 1) # (B, C, H, W) -> (B, H*W, C)

        
        # 2. 运行Transformer精炼头
        refined_logits, refined_bbox_deltas = self.refiner_head(object_features, memory)
        
        
        # --- 整合与返回 ---
        if self.training:
            dummy_loss = torch.tensor(1.0, requires_grad=True, device=x.device)
            # 你需要将上面的真实loss计算逻辑替换掉这个虚拟loss
            return dummy_loss
            
        else:
            # 在推理模式下，返回最终的精炼结果
            return refined_logits, refined_bbox_deltas


if __name__ == '__main__':
    # 这是一个简单的测试，确保模型可以被实例化并运行
    # 你需要下载预训练的yolov9-c.pt权重文件
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        
    # 确保你已经下载了yolov9-c.pt权重文件
    # pip install yolov9
    try:
        model = HiDefYOLO(yolov9_variant='yolov9-c.pt').to(device)
        model.eval()
        
        # 创建一个随机的输入图像
        dummy_input = torch.rand(1, 3, 640, 640).to(device)
        
        # 运行前向传播
        with torch.no_grad():
            logits, deltas = model(dummy_input)
        
        print("\n✅ Hi-Def-YOLO model forward pass successful!")
        print(f"   - Output Logits Shape: {logits.shape}")
        print(f"   - Output Bbox Deltas Shape: {deltas.shape}")

    except Exception as e:
        print(f"\n❌ Error during model instantiation or forward pass: {e}")
        print("   - Please make sure you have 'yolov9-c.pt' in your working directory.")
        print("   - You can download it from the official YOLOv9 repository.")
