# file: models/transformer_head.py

import torch
import torch.nn as nn

class TransformerRefinerHead(nn.Module):
    """
    一个基于Transformer的检测头，用于精炼和修正检测结果。
    它接收一组目标特征，通过自注意力机制学习它们之间的全局关系。
    """
    def __init__(self, feature_dim, num_heads=8, num_layers=2):
        """
        Args:
            feature_dim (int): 输入特征向量的维度.
            num_heads (int): 多头注意力机制的头数.
            num_layers (int): Transformer Decoder层的数量.
        """
        super().__init__()
        self.feature_dim = feature_dim
        
        # 定义一个标准的Transformer Decoder层
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            batch_first=True # !!重要!!: 输入数据的格式是 (batch, seq, feature)
        )
        
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers
        )
        
        # 输出层，用于预测最终的类别和边界框修正量
        self.class_head = nn.Linear(feature_dim, 80) # 假设COCO数据集80类
        self.bbox_head = nn.Linear(feature_dim, 4)   # 预测(dx, dy, dw, dh)修正量
        
        print(f"✅ TransformerRefinerHead initialized.")
        print(f"   - Feature Dimension: {feature_dim}")
        print(f"   - Attention Heads: {num_heads}")
        print(f"   - Decoder Layers: {num_layers}")

    def forward(self, object_features, memory):
        """
        Args:
            object_features (Tensor): 从候选框中提取的特征向量. 
                                      Shape: (batch_size, num_objects, feature_dim)
            memory (Tensor): 来自骨干网络的高层语义信息，作为Transformer的memory输入.
                             Shape: (batch_size, num_patches, feature_dim)
        
        Returns:
            Tuple[Tensor, Tensor]: 预测的类别和边界框修正量.
        """
        # Transformer Decoder需要一个memory输入，这里我们用object_features自身作为memory的简化
        # 在更复杂的实现中，memory可以来自骨干网络的其他部分
        refined_features = self.transformer_decoder(object_features, memory)
        
        # 预测最终结果
        pred_logits = self.class_head(refined_features)
        pred_bbox_deltas = self.bbox_head(refined_features)
        
        return pred_logits, pred_bbox_deltas