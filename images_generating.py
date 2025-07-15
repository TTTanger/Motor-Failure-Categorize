import pandas as pd
import os
import numpy as np


from PIL import Image


# 读取CSV文件
def read_csv(file_path):
    df = pd.read_csv(file_path, skiprows=0)
    return df


def process_data(data):
    # 复制最后一行数据
    last_row = data.iloc[-1, :].copy()
    last_row_df = last_row.to_frame().T  # 将 Series 转换为 DataFrame

    # 在原始数据的开始和结束分别追加最后一行数据
    df = pd.concat([last_row_df, data, last_row_df], ignore_index=True)

    # 创建一个新的空 DataFrame，列名与 data 相同
    new_data = pd.DataFrame(columns=data.columns)

    # 应用滑动窗口处理逻辑
    for col in df.columns:
        for i in range(1, len(df) - 1):
            new_data.at[i - 1, col] = (df.at[i, col] ** 2) - (
                df.at[i - 1, col] * df.at[i + 1, col]
            )

    # 返回处理后的结果
    return new_data


# 处理指定路径下的所有CSV文件
def process_files_in_directory(directory_path):
    if not os.path.exists(directory_path):
        print(f"The directory does not exist: {directory_path}")
        return

    new_files_path = os.path.join(directory_path, "buffer")
    if not os.path.exists(new_files_path):
        os.makedirs(new_files_path)  # 创建目录

    for filename in os.listdir(directory_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(directory_path, filename)
            print(f"Processing file: {filename}")

            # 读取数据
            data = read_csv(file_path)

            # 处理数据
            new_data = process_data(data)

            # 写入新文件，确保不写入索引
            new_file_path = os.path.join(new_files_path, filename)
            new_data.to_csv(new_file_path, index=False)
            print(f"Processed file saved as: {new_file_path}")


directory_path = 'workplace/sorted_data/TKEO_origin'
process_files_in_directory(directory_path)

# 定义原始文件夹路径和目标文件夹路径
original_folder = 'workplace/sorted_data/TKEO_origin/buffer'
target_folder = 'workplace/sorted_data/TKEO_normalized'

# 确保目标文件夹存在
if not os.path.exists(target_folder):
    os.makedirs(target_folder)

# 遍历原始文件夹中的所有CSV文件
for filename in os.listdir(original_folder):
    if filename.endswith('.csv'):
        # 构造文件的完整路径
        file_path = os.path.join(original_folder, filename)

        # 使用pandas读取CSV文件
        df = pd.read_csv(file_path)

        # 计算每一列的平均值
        columns_avg = df.mean()

        # 对前3列进行归一化处理
        for i in range(3):  # 只对前3列进行操作
            if not pd.isnull(columns_avg.iloc[i]):  # 确保平均值不是 NaN
                df.iloc[:, i] = (df.iloc[:, i] - columns_avg.iloc[i]) / columns_avg.iloc[i]

        # 构造新文件的完整路径
        new_file_path = os.path.join(target_folder, filename)

        # 将归一化后的数据保存到新的CSV文件
        df.to_csv(new_file_path, index=False)

print("归一化数据保存完成")


# 之字形排列函数
def zigzag(data, rows, cols):
    if len(data) != rows * cols:
        raise ValueError("数据长度必须等于行数乘以列数")

        # 将一维数组转换为NumPy数组
    data = np.array(data)

    # 初始化二维数组
    zigzag_matrix = np.zeros((rows, cols), dtype=data.dtype)

    # 之字形排列
    index = 0
    for i in range(rows):
        if i % 2 == 0:  # 偶数行，从左到右填充
            for j in range(cols):
                zigzag_matrix[i, j] = data[index]*255
                index += 1
        else:  # 奇数行，从右到左填充
            for j in range(cols - 1, -1, -1):
                zigzag_matrix[i, j] = data[index]*255
                index += 1

    return zigzag_matrix


# 定义文件夹路径
folder_path = 'workplace/sorted_data/TKEO_normalized'
target_path = 'workplace/sorted_data/matrix'

# 确保目标文件夹存在
if not os.path.exists(target_path):
    os.makedirs(target_path)

# 定义矩阵的维度
matrix_rows, matrix_columns = 60, 100

# 遍历文件夹中的所有CSV文件
for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(folder_path, filename)
        df = pd.read_csv(file_path)
        # 直接将列作为数组传递给zigzag函数
        Ia_array = zigzag(df.iloc[:, 0].values, matrix_rows, matrix_columns)
        Ib_array = zigzag(df.iloc[:, 1].values, matrix_rows, matrix_columns)
        Ic_array = zigzag(df.iloc[:, 2].values, matrix_rows, matrix_columns)

        # 保存矩阵到文件
        np.save(os.path.join(target_path, f'{os.path.splitext(filename)[0]}_Ia.npy'), Ia_array)
        np.save(os.path.join(target_path, f'{os.path.splitext(filename)[0]}_Ib.npy'), Ib_array)
        np.save(os.path.join(target_path, f'{os.path.splitext(filename)[0]}_Ic.npy'), Ic_array)
        print(f"Zigzag matrices for {filename} have been created and saved.")

print("Matrix Processing complete, starting generating images.")

# 指定.npy文件所在的目录
npy_files_directory = 'workplace/sorted_data/matrix'

# 指定图像保存的目录
save_directory = 'workplace/result/imgs'
os.makedirs(save_directory, exist_ok=True)

# 获取目录中所有.npy文件的列表，只获取一个通道的文件
# 只处理红通道的文件
npy_files = [f for f in os.listdir(npy_files_directory) if f.endswith('_Ia.npy')]

# 遍历所有红通道的.npy文件
for npy_file in npy_files:
    prefix = npy_file.split('_Ia')[0]

    # 构造绿和蓝通道的文件名
    green_filename = f"{prefix}_Ib.npy"
    blue_filename = f"{prefix}_Ic.npy"

    # 检查绿和蓝通道的文件是否存在
    green_file_path = os.path.join(npy_files_directory, green_filename)
    blue_file_path = os.path.join(npy_files_directory, blue_filename)
    if not os.path.isfile(green_file_path) or not os.path.isfile(blue_file_path):
        print(f"Missing green or blue channel for prefix {prefix}")
        continue  # 如果有缺失的通道，跳过当前循环

    # 加载所有通道
    r_channel = np.load(os.path.join(npy_files_directory, npy_file))
    g_channel = np.load(green_file_path)
    b_channel = np.load(blue_file_path)

    # 合并三个通道为一个图像
    image_array = np.stack([r_channel, g_channel, b_channel], axis=-1).astype(np.uint8)

    # 使用numpy数组创建PIL图像
    image = Image.fromarray(image_array, 'RGB')

    # 在保存图像之前，调整图像大小
    image = image.resize((256, 256), Image.LANCZOS)

    # 保存调整大小后的图像
    image.save(os.path.join(save_directory, f"{prefix}.png"))
    print(f"Image saved as {os.path.join(save_directory, f'{prefix}.png')}")

print("Images generating complete.")
