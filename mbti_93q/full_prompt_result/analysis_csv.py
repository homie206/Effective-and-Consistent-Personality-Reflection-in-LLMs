import pandas as pd
import os

# 要处理的目录
input_directory = '/home/hmsun/llama3/93q_0818'
output_directory = '/home/hmsun/llama3/93q_0818/processed'

# 创建输出目录（如果不存在）
os.makedirs(output_directory, exist_ok=True)

# 遍历目录中的每个文件
for filename in os.listdir(input_directory):
    if filename.endswith('.csv'):
        input_file = os.path.join(input_directory, filename)
        data = pd.read_csv(input_file)

        # 计算第 2-9 列的总和
        sums = data.iloc[:, 1:9].sum().tolist()

        # 统计第 10 列的类型数量
        type_counts = data.iloc[:, 9].value_counts().to_dict()

        # 创建一个新的 DataFrame 用于记录总和
        sums_df = pd.DataFrame([sums], columns=data.columns[1:9])

        # 将总和 DataFrame 连接到原 DataFrame
        result = pd.concat([data, sums_df], ignore_index=True)

        # 写入处理后的数据到新的 CSV 文件
        output_file = os.path.join(output_directory, filename)
        result.to_csv(output_file, index=False)

        # 生成同名 TXT 文件作为记录
        txt_file = os.path.join(output_directory, f'{os.path.splitext(filename)[0]}.txt')
        with open(txt_file, 'w') as f:
            # 写入第 2-9 列的总和
            f.write('总和:\n')
            f.write(', '.join(map(str, sums)) + '\n\n')
            # 写入第 10 列的类型和计数
            f.write('第 10 列类型计数:\n')
            for key, value in type_counts.items():
                f.write(f'{key}: {value}\n')

print('所有 CSV 文件已处理，总和和类型计数已记录。')
