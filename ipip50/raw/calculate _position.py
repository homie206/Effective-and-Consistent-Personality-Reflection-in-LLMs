import pandas as pd
import numpy as np

# 读取数据
df = pd.read_csv('/home/hmsun/IPIP-FFM-data-8Nov2018/data-final.csv', sep='\t')
df2 = pd.read_csv('/home/hmsun/llama3/ipip_50/result/test_score_with_final.csv')

# 提取需要的列
dims = ['EXT', 'EST', 'AGR', 'CSN', 'OPN']
columns = [i + str(j) for j in range(1, 11) for i in dims]
df = df[columns]


def get_final_scores(columns, dim):
    score = 0
    if dim == 'EXT':
        score += columns[0]
        score -= columns[1]
        score += columns[2]
        score -= columns[3]
        score += columns[4]
        score -= columns[5]
        score += columns[6]
        score -= columns[7]
        score += columns[8]
        score -= columns[9]
    if dim == 'EST':
        score -= columns[0]
        score += columns[1]
        score -= columns[2]
        score += columns[3]
        score -= columns[4]
        score -= columns[5]
        score -= columns[6]
        score -= columns[7]
        score -= columns[8]
        score -= columns[9]
    if dim == 'AGR':
        score -= columns[0]
        score += columns[1]
        score -= columns[2]
        score += columns[3]
        score -= columns[4]
        score += columns[5]
        score -= columns[6]
        score += columns[7]
        score += columns[8]
        score += columns[9]
    if dim == 'CSN':
        score += columns[0]
        score -= columns[1]
        score += columns[2]
        score -= columns[3]
        score += columns[4]
        score -= columns[5]
        score += columns[6]
        score -= columns[7]
        score += columns[8]
        score += columns[9]
    if dim == 'OPN':
        score += columns[0]
        score -= columns[1]
        score += columns[2]
        score -= columns[3]
        score += columns[4]
        score -= columns[5]
        score += columns[6]
        score += columns[7]
        score += columns[8]
        score += columns[9]
    return score


# 计算每个维度的总分
for dim in dims:
    df[dim + '_all'] = df.apply(lambda r: get_final_scores([r[dim + str(j)] for j in range(1, 11)], dim), axis=1)


def cal_test_position(test_score, df):
    positions = {}
    cnt = 0
    for dim in dims:
        df_tmp = df.sort_values(by=dim + '_all')
        target_value = test_score[cnt]
        # 获取目标值的索引位置
        if target_value in df_tmp[dim + '_all'].values:
            index_position = df_tmp[dim + '_all'][df_tmp[dim + '_all'] == target_value].index[0]
            percentage_position = (index_position + 1) / len(df_tmp[dim + '_all']) * 100
            positions[dim + '_position'] = percentage_position
        else:
            positions[dim + '_position'] = None  # 如果目标值不在数据中
        cnt += 1
    return positions


# 遍历 df2 的每一行，并将位置添加到 df2
for index, row in df2.iterrows():
    test_score = row[['EXT_Score', 'EST_Score', 'AGR_Score', 'CSN_Score', 'OPN_Score']].values.tolist()
    positions = cal_test_position(test_score, df)

    # 将位置添加到 df2
    for key, value in positions.items():
        df2.at[index, key] = value

# 保存更新后的 df2
df2.to_csv('/home/hmsun/llama3/ipip_50/result/updated_test_scores.csv', index=False)

# 可选：查看更新后的 df2
print(df2.head())
