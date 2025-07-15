import torch
import os

from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from cnn import AdvancedCNNModelWithSE


class CustomDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        self.labels = self._load_labels()

    def __len__(self):
        return len(self.image_files)

    def _load_labels(self):
        labels = []
        for img_name in self.image_files:
            label_name = img_name.replace('.png', '.txt').replace('.jpg', '.txt').replace('.jpeg', '.txt')
            label_path = os.path.join(self.label_dir, label_name)
            with open(label_path, 'r') as f:
                label = f.read().strip()
                if label not in ['normal', 'abnormal']:
                    raise ValueError(f"Label '{label}' is not recognized.")
                labels.append(label)
        return labels

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('L')
        label_str = self.labels[idx]
        label_map = {'normal': 0, 'abnormal': 1}
        label = label_map[label_str]

        if self.transform:
            image = self.transform(image)

        return image, label

    def get_all_labels(self):
        return [self.labels[idx] for idx in range(len(self.image_files))]


# 定义图像预处理流程
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(),  # 水平翻转
    transforms.RandomRotation(10),  # 随机旋转
    transforms.ToTensor(),
    transforms.Normalize((0.5, ), (0.5, ))
])

# 创建验证数据集和数据加载器
validation_dataset = CustomDataset(
    image_dir='dataset/test/imgs',
    label_dir='dataset/test/labels',
    transform=transform
)

validation_loader = DataLoader(
    dataset=validation_dataset,
    batch_size=32,
    shuffle=False
)


# 加载模型
model = AdvancedCNNModelWithSE()
model.load_state_dict(torch.load('model_best.pth'))  # 确保文件路径正确
model.eval()


# 定义模型验证函数
def validate_model(model, validation_loader):
    correct = 0
    with torch.no_grad():
        total = len(validation_dataset)  # 直接使用验证集的总样本数
        all_predicted = []
        all_labels = []
        for images, labels in validation_loader:
            outputs = model(images)
            # 确保outputs是一维的，并且squeeze去除单维度
            predicted = torch.round(torch.sigmoid(outputs.squeeze()))
            correct += (predicted == labels.squeeze()).sum().item()
            all_predicted.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    # 计算混淆矩阵
    conf_matrix_fine_tuned = confusion_matrix(all_labels, all_predicted)
    print("Confusion Matrix after fine-tuning:\n", conf_matrix_fine_tuned)
    validation_accuracy = 100 * correct / total
    print(f"Validation Accuracy: {validation_accuracy:.2f}%")
    return validation_accuracy


# 调用验证函数
validate_model(model, validation_loader)
