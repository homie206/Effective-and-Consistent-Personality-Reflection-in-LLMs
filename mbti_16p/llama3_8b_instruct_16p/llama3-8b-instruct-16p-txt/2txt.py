import os
import pandas as pd
from ast import literal_eval  # 导入 literal_eval

# 指定文件夹路径
folder_path = '/home/hmsun/llama3/result_16p'
output_path = '/home/hmsun/llama3/llama3-1-8b-16p-txt'

# 确保输出路径存在
os.makedirs(output_path, exist_ok=True)

# 遍历文件夹中的所有 CSV 文件
for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        csv_path = os.path.join(folder_path, filename)

        # 读取 CSV 文件
        df = pd.read_csv(csv_path)

        # 确保 "Values" 或 "values" 和 "Code" 列存在
        values_column = df.get('Values') if 'Values' in df.columns else df.get('values')
        code_column = df.get('Code')

        if values_column is not None and code_column is not None:
            sums = [0] * 5
            count = 0
            code_counts = {}

            for index, row in df.iterrows():
                try:
                    # 使用 literal_eval 而不是 eval
                    values = literal_eval(row[values_column.name])  # 将字符串转为列表
                    sums = [s + v for s, v in zip(sums, values)]
                    count += 1

                    # 统计“Code”列
                    code = row[code_column.name]
                    code_counts[code] = code_counts.get(code, 0) + 1

                except Exception as e:
                    print(f"处理索引 {index} 时出错: {e}")

            # 计算平均数
            averages = [s / count for s in sums] if count > 0 else [0] * 5

            # 生成输出 TXT 文件
            output_txt = os.path.join(output_path, f'statistics_{filename[:-4]}.txt')
            with open(output_txt, 'w') as f:
                f.write("总和:\n")
                f.write(f"E: {sums[0]}, N: {sums[1]}, T: {sums[2]}, J: {sums[3]}, A: {sums[4]}\n")
                f.write("平均数:\n")
                f.write(f"E: {averages[0]}, N: {averages[1]}, T: {averages[2]}, J: {averages[3]}, A: {averages[4]}\n")
                f.write("Code列计数:\n")
                for code, count in code_counts.items():
                    f.write(f"{code}: {count}\n")

print("统计完成！")