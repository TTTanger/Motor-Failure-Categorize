import os
import torch
from sklearn.metrics import confusion_matrix
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CustomBinaryDataset(Dataset):
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
            image = image.to(device, dtype=torch.float32)  # Move tensor to GPU and correct dtype

        return image, label

    def get_all_labels(self):
        return [self.labels[idx] for idx in range(len(self.image_files))]


# 定义SE块
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.avg_pool(x).view(b, c)
        avg_out = self.fc(avg_out).view(b, c, 1, 1)
        return x * avg_out.expand_as(x)


# 定义带有SE块的CNN模型
class AdvancedCNNModelWithSE(nn.Module):
    def __init__(self):
        super(AdvancedCNNModelWithSE, self).__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2),
            SELayer(8)  # 添加SE块
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(8, 21, 3, padding=1),
            nn.BatchNorm2d(21),
            nn.ReLU(),
            nn.MaxPool2d(2),
            SELayer(21)  # 添加SE块
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(21, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            SELayer(32)  # 添加SE块
        )
        self.fc1 = nn.Linear(32 * 8 * 8, 512)
        self.dropout = nn.Dropout(0.5).to(device)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)  # 二元分类，输出一个值

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = x.view(x.size(0), -1)  # 确保这里与全连接层的输入匹配
        x = F.relu(self.fc1(x))
        x = self.dropout(x) if self.training else x  # 仅在训练时应用Dropout
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def adjust_learning_rate(optimizer, epoch, initial_lr=0.001, decay_rate=0.1, decay_step=5):
    lr = initial_lr * (decay_rate ** (epoch // decay_step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr  # 衰减学习率


def count_class_in_dataset(dataset, class_name):
    class_count = 0
    label_map = {'normal': 0, 'abnormal': 1}
    # 确保class_name是正确的类别名称
    if class_name not in label_map:
        raise ValueError(f"Class name {class_name} not recognized.")

    # 迭代数据集中的所有项目来计算特定类别的数量
    for _, label in dataset:
        if label == label_map[class_name]:
            class_count += 1
    return class_count


if __name__ == "__main__":

    # 设置模型输出路径
    export_path = 'model'

    # 检查路径是否存在
    if not os.path.exists(export_path):
        print(f"路径 {export_path} 不存在。尝试创建...")
        try:
            os.makedirs(export_path)
            print(f"路径 {export_path} 已创建。")
        except PermissionError:
            print(f"没有权限在 {export_path} 创建目录。")
    else:
        print(f"路径 {export_path} 已存在。")

    # 检查写权限
    if os.access(export_path, os.W_OK):
        print(f"当前用户在 {export_path} 有写权限。")
    else:
        print(f"当前用户在 {export_path} 没有写权限。")

    # 定义图像预处理流程
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(),  # 水平翻转
        transforms.RandomRotation(10),  # 随机旋转
        transforms.ToTensor(),
        transforms.Normalize((0.5, ), (0.5, ))
    ])

    # 预训练目录
    pre_image_dir = 'dataset/pre_training_dataset/imgs'
    pre_label_dir = 'dataset/pre_training_dataset/labels'

    # 预训练数据集
    pre_dataset = CustomBinaryDataset(image_dir=pre_image_dir, label_dir=pre_label_dir, transform=transform)

    # 获取所有标签并转换为NumPy数组
    pre_labels = np.array(pre_dataset.get_all_labels())

    # 为每个类别创建索引列表
    normal_indices = np.where(pre_labels == 'normal')[0]
    abnormal_indices = np.where(pre_labels == 'abnormal')[0]

    # 计算训练集和验证集的样本数量
    total_samples = len(pre_dataset)
    train_samples = int(total_samples * 0.8)  # 训练集占80%
    val_samples = total_samples - train_samples  # 验证集占20%

    # 打乱索引
    np.random.shuffle(normal_indices)
    np.random.shuffle(abnormal_indices)

    # 从每个类别中选择一半的样本作为训练集，确保训练集中的比例是1:1
    train_normal_indices = normal_indices[:train_samples // 2]
    train_abnormal_indices = abnormal_indices[:train_samples // 2]

    # 剩余的样本作为验证集
    val_normal_indices = normal_indices[train_samples // 2:]
    val_abnormal_indices = abnormal_indices[train_samples // 2:]

    # 合并训练集和验证集的索引
    train_indices = np.concatenate([train_normal_indices, train_abnormal_indices])
    val_indices = np.concatenate([val_normal_indices, val_abnormal_indices])

    # 使用Subset来创建训练集和验证集
    train_dataset = Subset(pre_dataset, train_indices)
    val_dataset = Subset(pre_dataset, val_indices)

    # 检查划分结果是否满足条件
    print(f'Total dataset size: {total_samples}')
    print(f'Training set size: {len(train_dataset)} (80% of total)')
    print(f'Validation set size: {len(val_dataset)} (20% of total)')
    print(f'Training set normal count: {len(train_normal_indices)}')
    print(f'Training set abnormal count: {len(train_abnormal_indices)}')

    # 创建训练集和验证集的DataLoader
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    normal_count = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)  # 将数据和标签移动到GPU
        # 将标签张量转换为numpy数组进行迭代
        label_array = labels.cpu().numpy()
        # 累加标签为0的计数
        normal_count += np.count_nonzero(label_array == 0)

    normal_count = 0
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)  # 将数据和标签移动到GPU
        # 将标签张量转换为numpy数组进行迭代
        label_array = labels.cpu().numpy()
        # 累加标签为0的计数
        normal_count += np.count_nonzero(label_array == 0)

    # 实例化模型
    model = AdvancedCNNModelWithSE().to(device)

    # 定义损失函数和优化器
    criterion = nn.BCEWithLogitsLoss().to(device)
    pre_optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 20

    # 早停参数
    patience = 5  # 没有改善时等待的epoch数
    best_val_loss = float('inf')  # 最佳验证损失
    epochs_no_improve = 0  # 没有改善的连续epoch数
    early_stopping_rounds = 0  # 早停的轮数
    # 定义验证频率
    validation_frequency = 1  # 每个epoch结束时进行验证



    # 训练与验证循环
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            pre_optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs.squeeze(), labels.float())
            loss.backward()
            pre_optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}')

        # 验证
        if epoch % validation_frequency == 0 or epoch == num_epochs - 1:
            model.eval()
            val_loss = 0
            correct = 0
            total = len(val_dataset)
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs.squeeze(), labels.float())
                    val_loss += loss.item()
                    predicted = torch.round(torch.sigmoid(outputs.squeeze()))
                    correct += (predicted == labels.squeeze()).sum().item()

            val_loss /= len(val_loader)
            accuracy = correct / total
            print(f'Validation Loss: {val_loss:.4f}, Accuracy: {100 * accuracy:.2f}%')

            # 更新最佳验证损失和早停逻辑
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                # 保存最佳模型状态
                torch.save(model.state_dict(), 'model/model_pre.pth')
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f'Early stopping initiated: No improvement after {patience} epochs.')
                    early_stopping_rounds = epoch
                    break

    # 如果早停被触发，加载最佳模型权重
    if early_stopping_rounds > 0:
        model.load_state_dict(torch.load('model/model_pre.pth'))
        print(f'Model weights loaded from epoch {early_stopping_rounds}.')

    # 最后评估模型性能
    model.eval()
    with torch.no_grad():
        correct = 0
        total = len(val_dataset)  # 直接使用验证集的总样本数
        all_predicted = []
        all_labels = []
        for images, labels in val_loader:
            outputs = model(images)
            # 确保outputs是一维的，并且squeeze去除单维度
            predicted = torch.round(torch.sigmoid(outputs.squeeze()))
            correct += (predicted.to(device) == labels.squeeze().to(device)).sum().item()
            all_predicted.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 将标签转换为整数
    all_predicted = np.array(all_predicted).astype(int)
    all_labels = np.array(all_labels).astype(int)

    # 计算混淆矩阵
    conf_matrix = confusion_matrix(all_labels, all_predicted)
    print("Confusion Matrix:\n", conf_matrix)

    # 计算并打印准确度
    accuracy = correct / total
    print(f'Accuracy of the model on the validation images: {100 * accuracy:.2f}%')

    input_size = (1, 1, 64, 64)

    '''微调开始'''
    # 加载预训练模型权重
    model.load_state_dict(torch.load('model/model_pre.pth'))

    '''根据路径修改'''
    # 创建微调数据集（只训练，验证用原来的数据集）
    fine_tuning_dataset = CustomBinaryDataset(
        image_dir='dataset/fine_tuning_dataset/imgs',
        label_dir='dataset/fine_tuning_dataset/labels',
        transform=transform
    )

    # 创建微调数据加载器
    fine_tuning_loader = DataLoader(fine_tuning_dataset, batch_size=16, shuffle=True)

    # 设置微调的优化器
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)  # 只优化需要训练的参数

    fine_tuning_num_epochs = 10

    # 早停参数
    patience = 3   # 没有改善时等待的epoch数
    best_val_loss = float('inf')  # 最佳验证损失
    epochs_no_improve = 0  # 没有改善的连续epoch数
    early_stopping_rounds = 0  # 早停的轮数
    # 定义验证频率
    validation_frequency = 1  # 每个epoch结束时进行验证

    # 微调模型
    # 训练与验证循环
    for epoch in range(fine_tuning_num_epochs):
        model.train()
        train_loss = 0
        for images, labels in fine_tuning_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs.squeeze(), labels.float())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(fine_tuning_loader)
        print(f'Epoch [{epoch+1}/{fine_tuning_num_epochs}], Train Loss: {train_loss:.4f}')

        # 验证
        if epoch % validation_frequency == 0 or epoch == num_epochs - 1:
            model.eval()
            val_loss = 0
            correct = 0
            total = len(val_dataset)
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs.squeeze(), labels.float())
                    val_loss += loss.item()
                    predicted = torch.round(torch.sigmoid(outputs.squeeze()))
                    correct += (predicted == labels.squeeze()).sum().item()

            val_loss /= len(val_loader)
            accuracy = correct / total
            print(f'Validation Loss: {val_loss:.4f}, Accuracy: {100 * accuracy:.2f}%')

            # 更新最佳验证损失和早停逻辑
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                # 保存最佳模型状态
                torch.save(model.state_dict(), 'model/target_model.pth')
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f'Early stopping initiated: No improvement after {patience} epochs.')
                    early_stopping_rounds = epoch
                    break

    # 如果早停被触发，加载最佳模型权重
    if early_stopping_rounds > 0:
        model.load_state_dict(torch.load('model/target_model.pth'))
        print(f'Model weights loaded from epoch {early_stopping_rounds}.')

    # 最后评估模型性能
    model.eval()
    with torch.no_grad():
        correct = 0
        total = len(val_dataset)  # 直接使用验证集的总样本数
        all_predicted = []
        all_labels = []
        for images, labels in val_loader:
            outputs = model(images)
            # 确保outputs是一维的，并且squeeze去除单维度
            predicted = torch.round(torch.sigmoid(outputs.squeeze()))
            correct += (predicted.to(device) == labels.squeeze().to(device)).sum().item()
            all_predicted.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 计算混淆矩阵
    conf_matrix_fine_tuned = confusion_matrix(all_labels, all_predicted)
    print("Confusion Matrix after fine-tuning:\n", conf_matrix_fine_tuned)
    # 计算并打印准确度
    accuracy = correct / total
    print(f'Accuracy of the model on the validation images: {100 * accuracy:.2f}%')

    # 保存微调后的模型
    torch.save(model.state_dict(), 'model/target_model.pth')
    pass
