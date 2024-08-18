import json
import torch
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer
import transformers

mbti_questions = json.load(
    open('/home/hmsun/llama3/py/mbti_questions_en.json', 'r', encoding='utf8')
)

OUTPUT_PATH = '/home/hmsun/llama3/res/mbti-llama3.1-8b-instruct-output.txt'
RESULT_PATH = '/home/hmsun/llama3/res/mbti-llama3.1-8b-instruct-result.csv'

counter = {
    'E': 0,
    'I': 0,
    'N': 0,
    'S': 0,
    'F': 0,
    'T': 0,
    'J': 0,
    'P': 0
}

def get_model_examing_result(model_id):
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )

    results = []

    for cycle in tqdm(range(100)):
        with open(OUTPUT_PATH, 'a', encoding='utf-8') as f:
            f.write(f"\n\n====== Cycle {cycle+1} ======\n\n")

            for question, options in mbti_questions.items():

                messages = [
                    {"role": "system", "content": "Answer the question, directly output A or B."},
                    {"role": "user", "content": options["question"]}
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

                generated_text = outputs[0]["generated_text"]
                f.write(f"Question: {options['question']}\n")
                f.write(f"generated_text: {generated_text}\n\n")

                answer00 = generated_text[-1]["content"]
                f.write(f"raw_answer: {answer00}\n\n")

                parsed_result = re.search(r"[abAB][^a-zA-Z]", answer00[:6])
                if parsed_result or (answer00 == "A" or answer00 == "B"):
                    if parsed_result:
                       answer = parsed_result.group()[0].upper()
                       f.write(f"answer: {answer}\n\n")
                    if answer00 == "A" or answer00 == "B":
                        answer = answer00
                else:
                    answer = answer00
                    f.write(f"No suitable answer: {answer00}\n\n")

                if answer == "A":
                        counter[options["A"]] += 1
                        f.write(f"options: {options['A']}\n\n")
                elif answer == "B":
                        counter[options["B"]] += 1
                        f.write(f"options: {options['B']}\n\n")
                else:
                    f.write(f"options: (no options available)\n\n")

        EI = 'E' if counter['E'] > counter['I'] else 'I'
        NS = 'N' if counter['N'] > counter['S'] else 'S'
        FT = 'F' if counter['F'] > counter['T'] else 'T'
        JP = 'J' if counter['J'] > counter['P'] else 'P'

        result = {
            "num_of_cycles": cycle+1,
            "E": counter['E'],
            "I": counter['I'],
            "N": counter['N'],
            "S": counter['S'],
            "F": counter['F'],
            "T": counter['T'],
            "J": counter['J'],
            "P": counter['P'],
            "MBTI": EI + NS + FT + JP
        }

        results.append(result)

        print(f"Cycle {cycle+1} - Counter: {counter} - MBTI: {EI + NS + FT + JP}")

        for key in counter:
            counter[key] = 0

    df = pd.DataFrame(results)
    df.to_csv(RESULT_PATH, index=False)

if __name__ == '__main__':
    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    get_model_examing_result(model_id)
