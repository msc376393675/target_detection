# file: utils/datasets.py
import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import glob
import torchvision.transforms as T

class OPIXrayDataset(Dataset):
    def __init__(self, img_dir, label_dir, img_size=640):
        self.img_paths = sorted(glob.glob(os.path.join(img_dir, '*.jpg'))) # 假设是.jpg格式
        self.label_paths = sorted(glob.glob(os.path.join(label_dir, '*.txt')))
        self.img_size = img_size
        self.transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            # 在这里可以加入归一化等其他预处理
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label_path = self.label_paths[idx]

        # 加载图片
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        # 加载标签
        # 真实标签的加载会更复杂，需要处理padding，确保一个batch中标签数量一致
        # 这里做一个简化版
        labels = torch.zeros((100, 5)) # 假设最多100个目标
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                lines = f.readlines()
                for i, line in enumerate(lines):
                    if i >= 100: break
                    class_id, x, y, w, h = map(float, line.split())
                    labels[i, :] = torch.tensor([class_id, x, y, w, h])

        return image, labels
