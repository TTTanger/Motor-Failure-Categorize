import pandas as pd
import os


def process_simulation_data(csv_path, output_folder, sample_rate, sample_time):
    try:
        df = pd.read_csv(csv_path, skiprows=2)
        print("\n\nRead data from: {}".format(csv_path))
        
        expected_columns = ['IsLa [A]', 'IsLb [A]', 'IsLc [A]']
        if not set(expected_columns).issubset(df.columns):
            print("_________________________________________Error: Specified columns not found in data.")
            return

        df = df[expected_columns]
        print("Filtered data shape: {}".format(df.shape))

        step = 9 if sample_rate == 9009 else 1
        base_folder = os.path.basename(os.path.dirname(csv_path))
        base_filename = "{}.csv".format(base_folder)

        output_path1 = output_path2 = None
        if sample_time == 12:
            print('------------------------divide---------------------')
            mid_index = len(df) // 2
            df1 = df.iloc[:mid_index:step]
            df2 = df.iloc[mid_index:mid_index + mid_index:step]
            output_path1 = os.path.join(output_folder, "{}_1.csv".format(base_filename))
            output_path2 = os.path.join(output_folder, "{}_2.csv".format(base_filename))
            df1.to_csv(output_path1, index=False)
            df2.to_csv(output_path2, index=False)
            print("Processed files saved:\n{},\n********\n{}".format(output_path1, output_path2))
        else:
            df = df.iloc[::step]
            output_path = os.path.join(output_folder, base_filename)
            df.to_csv(output_path, index=False)
            print("Processed file saved: {}".format(output_path))
    except FileNotFoundError:
        print("File not found: {}".format(csv_path))
    except Exception as e:
        print("Error processing simulation data: {}".format(e))


def process_experiment_data(csv_file, output_folder):
    try:
        df = pd.read_csv(csv_file, skiprows=0)
        print("Read data from: {}".format(csv_file))
        
        expected_columns = ['Ia', 'Ib', 'Ic']
        if not set(expected_columns).issubset(df.columns):
            print("Error: Specified columns not found in data.")
            return

        df = df[expected_columns]
        print("Filtered data shape: {}".format(df.shape))

        # 计算后12秒的数据索引
        total_samples = len(df)
        start_index = int(total_samples - 12 * 50000)  # 转换为索引
        end_index = len(df)

        # 截取后12秒的数据
        df_12s = df.iloc[start_index:]

        # 重新采样到1 kHz
        df_resampled = df_12s.iloc[::50]  # 每50个样本取一个，相当于1 kHz采样率

        # 拆分为前后6秒的数据
        mid_index = len(df_resampled) // 2
        df1 = df_resampled.iloc[:mid_index]
        df2 = df_resampled.iloc[-mid_index:]

        # 获取上一级文件夹名
        folder_name = os.path.basename(os.path.dirname(csv_file))

        # 设置输出文件名
        base_filename = "{}_{}".format(folder_name, os.path.splitext(os.path.basename(csv_file))[0])
        output_path1 = os.path.join(output_folder, "{}_1.csv".format(base_filename))
        output_path2 = os.path.join(output_folder, "{}_2.csv".format(base_filename))

        # 设置列名
        df1.columns = ['IsLa [A]', 'IsLb [A]', 'IsLc [A]']
        df2.columns = ['IsLa [A]', 'IsLb [A]', 'IsLc [A]']

        # 保存文件
        df1.to_csv(output_path1, index=False)
        df2.to_csv(output_path2, index=False)
        print("Processed files saved: {}".format(output_path1))
        print("Processed files saved: {}".format(output_path2))
    except Exception as e:
        print("Error processing {} : {}".format(csv_file, e))


def main():
    xlsx_path = 'workplace\origin_data\数据说明.xlsx'

    # 输出文件夹
    output_folder = 'workplace\origin_data\pre_processed'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print("Created output folder: {}".format(output_folder))

    xl = pd.ExcelFile(xlsx_path)
    sheet = xl.parse(xl.sheet_names[0])

    for index, row in sheet.iterrows():
        file_name = row['文件名'].replace('/', '\\')
        print('\n\n',file_name)
        if row['数据类型'] == '仿真数据':
            csv_path = os.path.join('workplace\origin_data\仿真数据', file_name, 'cimtdtulos.csv')
            process_simulation_data(csv_path, output_folder, row['采样率(Hz)'], row['采样时间(s)'])
        elif row['数据类型'] == '试验数据':
            print('*****************************')
            csv_file = os.path.join('workplace\origin_data',file_name)
            process_experiment_data(csv_file, output_folder)
            print('***********************************')


main()
