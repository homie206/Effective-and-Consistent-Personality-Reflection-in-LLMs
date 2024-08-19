import pandas as pd
import os

# 定义文件夹路径
folder_path = '/home/hmsun/llama3/ipip_50/result2'

# 遍历文件夹中的所有 CSV 文件
for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(folder_path, filename)

        # 读取 CSV 文件
        df = pd.read_csv(file_path)

        # 计算平均位置
        mean_positions = df[['EXT_position', 'EST_position', 'AGR_position', 'CSN_position', 'OPN_position']].mean()

        # 添加高低列
        for col in ['EXT', 'EST', 'AGR', 'CSN', 'OPN']:
            df[f'{col}_high_or_low'] = df[f'{col}_position'].apply(lambda x: 'high' if x > 50 else 'low')

        # 统计高低数量
        high_low_counts = df[[f'{col}_high_or_low' for col in ['EXT', 'EST', 'AGR', 'CSN', 'OPN']]].apply(lambda x: x.value_counts(), axis=0).fillna(0)

        # 计算平均位置的高低
        mean_high_low = {col: 'high' if mean > 50 else 'low' for col, mean in mean_positions.items()}

        # 输出结果
        output_lines = ["平均位置：\n"]
        output_lines.append(mean_positions.to_string())
        output_lines.append("\n平均位置高低：\n")
        output_lines.append(str(mean_high_low))
        output_lines.append("\n高低统计：\n")
        output_lines.append(high_low_counts.to_string())

        # 生成同名 TXT 文件
        txt_file_path = os.path.join(folder_path, filename.replace('.csv', '_results.txt'))
        with open(txt_file_path, 'w') as f:
            f.write('\n'.join(output_lines))

        # 将高低列写回原 CSV 文件
        df.to_csv(file_path, index=False)

print("处理完成，所有结果已写入 TXT 文件和更新后的 CSV 文件。")
