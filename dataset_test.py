#!pip install torch transformers datasets
import torch
import re
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from collections import OrderedDict
# from datasets import load_dataset
import pandas as pd
from GPTZero_model_delPrint import GPT2PPL
# from model_1 import GPT2PPL
from sklearn.metrics import accuracy_score
import os

# 加载数据集
# dataset = load_dataset("artem9k/ai-text-detection-pile")
dataset = pd.read_parquet("../local_datasets/sampled_50_everySource.parquet")
dataset['text'] = dataset['text'].apply(lambda x: x.replace('\n', ''))

model = GPT2PPL()


# Step 4: Define a function to compute accuracy
def compute_accuracy(dataset):
    y_true = []
    y_pred = []
    count = 0
    for _,sample in dataset.iterrows():
        count += 1
        sentence = sample['text']
        target = sample['source']
        try:
            results,_ = model(sentence)
            if results == '小于100':
                continue
            y_true.append(target)
            y_pred.append(results['label'])
        except ValueError as e:
            print(f'Error processing sample {count}: {e}')
            continue
        # if count % 1000 == 0:
        #     print(f'已经测试{count}k个数据')
        print(count)
    return accuracy_score(y_true, y_pred)

# Step 5: Compute and print accuracy
accuracy = compute_accuracy(dataset)
print(f"Accuracy: {accuracy}")