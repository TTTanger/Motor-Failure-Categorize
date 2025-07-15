from torch.utils.data import Dataset
from PIL import Image
import os


class MyData(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.img_path = [f for f in os.listdir(self.root_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    def __getitem__(self, idx):
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.root_dir, img_name)
        img = Image.open(img_item_path).convert('RGB')  # 或者 'L' 如果是灰度图像

        # 使用 any 函数检查图像名称是否包含特定的关键字之一
        label = "abnormal" if any(
            keyword in img_name for keyword in ["broken", "fault", "r1b", "r2b", "r3b", "r4b"]) else "normal"

        return img, label

    def __len__(self):
        return len(self.img_path)


# todo 请根据自身需求替换该路径
# 数据集相关设置
root_dir = "workplace/result/imgs"  # 图片存放的根目录
label_dir = "workplace/result/labels"  # 标签存放的目录

# 创建标签目录
os.makedirs(label_dir, exist_ok=True)

# 实例化数据集
img_dataset = MyData(root_dir)

# 遍历数据集获取图像和标签，并将标签写入到文本文件中
for i, (img, label) in enumerate(img_dataset):
    file_name = os.path.splitext(img_dataset.img_path[i])[0]
    label_file_path = os.path.join(label_dir, f"{file_name}.txt")
    with open(label_file_path, 'w') as f:
        f.write(label)
