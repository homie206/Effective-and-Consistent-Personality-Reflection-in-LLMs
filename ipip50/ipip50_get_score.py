import json
import torch
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
import transformers
import re
import pandas as pd

# 创建列名列表
column_names = ['EXT1', 'AGR1', 'CSN1', 'EST1', 'OPN1',
                'EXT2', 'AGR2', 'CSN2', 'EST2', 'OPN2',
                'EXT3', 'AGR3', 'CSN3', 'EST3', 'OPN3',
                'EXT4', 'AGR4', 'CSN4', 'EST4', 'OPN4',
                'EXT5', 'AGR5', 'CSN5', 'EST5', 'OPN5',
                'EXT6', 'AGR6', 'CSN6', 'EST6', 'OPN6',
                'EXT7', 'AGR7', 'CSN7', 'EST7', 'OPN7',
                'EXT8', 'AGR8', 'CSN8', 'EST8', 'OPN8',
                'EXT9', 'AGR9', 'CSN9', 'EST9', 'OPN9',
                'EXT10', 'AGR10', 'CSN10', 'EST10', 'OPN10']

# 创建 DataFrame
df = pd.DataFrame(columns=column_names)

def extract_first_number(answer):
    match = re.search(r'^\d+', answer)
    if match:
        return int(match.group())
    else:
        return None

def get_response(q, model_id):
    messages = [
        {"role": "system", "content": '''Imagine you are a human. Given a statement of you. Please choose from the following options to identify how accurately this statement describes you. 
                        1. Very Accurate
                        2. Moderately Accurate
                        3. Neither Accurate Nor Inaccurate
                        4. Moderately Inaccurate
                        5. Very Inaccurate
                        Please only answer with the option number.'''},
        {"role": "user", "content": q}
    ]

    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = pipeline(
        messages,
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )

    generated_text = outputs[0]["generated_text"][-1]["content"]
    #     print('generated_text', generated_text)
    return generated_text


if __name__ == '__main__':
    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )


    with open('IPIP-50.txt', 'r') as f:
        question_list = f.readlines()
        answer_list = []
        extracted_numbers = []
        all_results = []

        for run in range(100):  # 运行100次
            extracted_numbers = []
            with open(f'result/llama3.1_raw_output_run{run + 1}.txt', 'w') as f:
                for q in question_list:
                    answer = get_response(q, model_id)
                    f.write(answer + '\n')
                    extracted_number = extract_first_number(answer)
                    extracted_numbers.append(extracted_number)

            print(f"Run {run + 1} extracted numbers:")
            print(extracted_numbers)

            all_results.append(extracted_numbers)

            # 将结果转换为 DataFrame
        result_df = pd.DataFrame(all_results, columns=column_names)

        # 保存结果到 CSV 文件
        result_df.to_csv('result/test_score_numbers.csv', index=False)




        df = pd.read_csv('result/test_score_numbers.csv', sep=',')
        df.head()

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


        for i in dims:
            df[i + '_all'] = df.apply(lambda r: get_final_scores(columns=[r[i + str(j)] for j in range(1, 11)], dim=i),
                                      axis=1)

        for i in ['EXT', 'EST', 'AGR', 'CSN', 'OPN']:
            print(f"{i}_all:")
            print(df[i + '_all'])
            print()

        final_scores = [df[i + '_all'][0] for i in dims]
        print(final_scores)

        for i in dims:
            df[i + '_Score'] = df.apply(
                lambda r: get_final_scores(columns=[r[i + str(j)] for j in range(1, 11)], dim=i), axis=1)

        # 保存结果到 CSV 文件
        original_df = pd.read_csv('/home/hmsun/llama3/ipip_50/result/test_score_numbers.csv', sep=',')

        # 合并新旧数据
        result_df = pd.concat([original_df, df[['EXT_Score', 'EST_Score', 'AGR_Score', 'CSN_Score', 'OPN_Score']]],
                              axis=1)
        # 保存结果到 CSV 文件
        result_df.to_csv('/home/hmsun/llama3/ipip_50/result/test_score_with_final.csv', index=False)


