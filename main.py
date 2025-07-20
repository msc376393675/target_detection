import torch
import argparse
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils.datasets import OPIXrayDataset
import os
import torch.nn.functional as F

from models.hidef_yolo import HiDefYOLO



def create_opixray_dataloader(img_dir, label_dir, batch_size, img_size=640):
    """真实的OPIXray数据加载器。"""
    dataset = OPIXrayDataset(img_dir, label_dir, img_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

def compute_loss(preds, targets):
    """一个简化的损失函数实现。"""
    refined_logits, refined_bbox_deltas = preds

    target_classes = targets[..., 0].long()
    target_boxes = targets[..., 1:]

    # 分类损失 (只对有目标的进行计算)
    valid_mask = target_classes > -1 # 假设-1是背景/padding
    loss_cls = F.cross_entropy(refined_logits[valid_mask], target_classes[valid_mask])

    # 回归损失 (只对有目标的进行计算)
    # TODO: refined_bbox_deltas需要转换成与target_boxes相同的格式
    loss_bbox = F.l1_loss(refined_bbox_deltas[valid_mask], target_boxes[valid_mask])

    return loss_cls + loss_bbox


def train_one_epoch(model, dataloader, optimizer, device, epoch, total_epochs):
    """训练一个周期的函数。"""
    model.train()  # 设置模型为训练模式
    total_loss = 0
    
    # 使用tqdm显示进度条
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{total_epochs} [Training]")
    for images, targets in pbar:
        images = images.to(device)
        targets = targets.to(device)
        
        # 清零梯度
        optimizer.zero_grad()
        
        # 前向传播
        # 假设模型在训练时直接返回损失
        loss = model(images, targets)
        
        
        # 反向传播
        loss.backward()
        
        # 更新权重
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")
        
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1} Training Average Loss: {avg_loss:.4f}")
    return avg_loss


@torch.no_grad()
def evaluate(model, dataloader, device):
    """评估模型的函数。"""
    model.eval()  # 设置模型为评估模式
    print("\nRunning validation...")
    
    for images, targets in tqdm(dataloader, desc="[Validation]"):
        images = images.to(device)
        # 在评估模式下，模型返回预测结果
        predictions = model(images)
        # TODO: 在这里处理predictions，并与targets比较，计算mAP等指标
        
    print("Validation finished. (mAP calculation logic needs to be implemented)")
    # return validation_metrics 
    return {"mAP": 0.5} # 返回一个虚拟的mAP


def main(args):
    """主执行函数。"""
    # 1. 设置设备 (CUDA or CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 2. 准备数据集
    print("⌛️ Loading OPIXray dataset...")
    train_img_dir = os.path.join(args.dataset_root, 'train/train_image')
    train_label_dir = os.path.join(args.dataset_root, 'train/train_annotation')
    train_loader = create_opixray_dataloader(train_img_dir, train_label_dir, args.batch_size)

    val_img_dir = os.path.join(args.dataset_root, 'test/test_image')
    val_label_dir = os.path.join(args.dataset_root, 'test/test_annotation')
    val_loader = create_opixray_dataloader(val_img_dir, val_label_dir, args.batch_size)
    print("✅ Dataset loaded.")
    
    # 3. 构建模型
    # HiDefYOLO将自动加载指定的YOLOv9和Real-ESRGAN权重
    model = HiDefYOLO(
        num_classes=args.num_classes,
        sr_in_channels=args.sr_channels,
        feature_dim=args.feature_dim,
        yolov9_variant=args.yolo_weights,
        sr_model_name=args.sr_weights # 
    ).to(device)

    # 4. 定义优化器和学习率调度器
    params_to_optimize = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params_to_optimize, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # 5. 开始训练循环
    print("\n🚀 Starting training for {} epochs...".format(args.epochs))
    for epoch in range(args.epochs):
        train_one_epoch(model, train_loader, optimizer, device, epoch, args.epochs)
        
        
        # 更新学习率
        scheduler.step()
        
        # TODO: 在这里添加保存模型的逻辑
        torch.save(model.state_dict(), f'hidef_yolo_epoch_{epoch+1}.pt')

    print("\n✅ Training complete!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hi-Def-YOLO Training Script')
    
    # 数据集和模型路径参数
    parser.add_argument('--data-path', type=str, default='./data/opixray.yaml', help='Path to the dataset config file')
    parser.add_argument('--yolo-weights', type=str, default='yolov9-c.pt', help='Path to YOLOv9 pretrained weights')
    parser.add_argument('--sr-weights', type=str, default='RealESRGAN_x4plus.pth', help='Filename of the Real-ESRGAN weights in the ./weights folder')

    # 训练超参数
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-2, help='Weight decay for optimizer')

    # 模型结构参数
    parser.add_argument('--num-classes', type=int, default=25, help='Number of classes in your dataset (e.g., OPIXray has 25)')
    parser.add_argument('--sr-channels', type=int, default=256, help='Number of input channels for the SR module')
    parser.add_argument('--feature-dim', type=int, default=256, help='Dimension of features for the Transformer head')

    args = parser.parse_args()
    
    # 确保权重文件夹存在
    if not Path('weights').exists():
        Path('weights').mkdir()
        
    main(args)
