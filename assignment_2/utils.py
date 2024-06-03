import os
import shutil
import pytesseract
import cv2

import matplotlib.pyplot as plt
import numpy as np

from jiwer import wer, cer
from tabulate import tabulate

def extract_text(img_dir, save_dir, config=None):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    else:
        shutil.rmtree(save_dir, ignore_errors=True)
        os.makedirs(save_dir)

    for i in os.listdir(img_dir):
        img = cv2.imread(os.path.join(img_dir, i))
        if config:
            text = pytesseract.image_to_string(img, lang='deu', config=config)
        else:
            text = pytesseract.image_to_string(img, lang='deu')
        img_idx = i.split(".")[0].split("_")[1]
        with open(os.path.join(save_dir, img_idx + ".txt"), "w") as f:
            f.write(text)

def evaluate(gt_path, et_path, print_results=True):
    word_error_rate = []
    char_error_rate = []

    gt_files = []
    for i in os.listdir(gt_path):
        gt_files.append(gt_path + "/" + i)

    et_files = []
    for i in os.listdir(et_path):
        et_files.append(et_path + "/" + i)

    for i in range(len(gt_files)):
        with open(gt_files[i], "r") as f:
            gt_text = f.read()
        with open(et_files[i], "r") as f:
            et_text = f.read()
        word_error_rate.append(wer(gt_text, et_text))
        char_error_rate.append(cer(gt_text, et_text))

    mean_wer = round(sum(word_error_rate)/len(word_error_rate), 2)
    mean_cer = round(sum(char_error_rate)/len(char_error_rate), 2)

    if print_results:
        print(f"Word Error Rate: {mean_wer}")
        print(f"Character Error Rate: {mean_cer}")

    return mean_wer, mean_cer, word_error_rate, char_error_rate

def plot_error_rates(word_error_rate, char_error_rate):
    fig, axs = plt.subplots(2, figsize=(15, 6))

    index = np.arange(len(word_error_rate)) + 1
    bar_width = 0.35
    opacity = 0.8
    axs[0].bar(index, word_error_rate, bar_width, alpha=opacity, color='blue', label='Word Error Rate')
    axs[0].bar(index + bar_width, char_error_rate, bar_width, alpha=opacity, color='red', label='Character Error Rate')
    axs[0].set_title('Non-Normalized Error Rates')
    axs[0].set_xlabel('Image Index')
    axs[0].set_ylabel('Error Rate')
    axs[0].set_xticks(index + bar_width / 2)
    axs[0].set_xticklabels([str(i) for i in index])
    axs[0].legend(loc='upper left')
    axs[0].set_xlim(0.5, len(word_error_rate) + 0.75)

    norm_word_error_rate = np.array(word_error_rate) / np.linalg.norm(word_error_rate)
    norm_char_error_rate = np.array(char_error_rate) / np.linalg.norm(char_error_rate)

    index = np.arange(len(norm_word_error_rate)) + 1
    axs[1].bar(index, norm_word_error_rate, bar_width, alpha=opacity, color='blue', label='Word Error Rate')
    axs[1].bar(index + bar_width, norm_char_error_rate, bar_width, alpha=opacity, color='red', label='Character Error Rate')
    axs[1].set_title('Normalized Error Rates')
    axs[1].set_xlabel('Image Index')
    axs[1].set_ylabel('Normalized Error Rate')
    axs[1].set_xticks(index + bar_width / 2)
    axs[1].set_xticklabels([str(i) for i in index])
    axs[1].legend(loc='upper left')
    axs[1].set_xlim(0.5, len(norm_word_error_rate) + 0.75)

    plt.tight_layout() 
    plt.show()

def put_res_to_table(description, wer, cer, print_table=True, table=None):
    if table is None:
        table = []

    table.append((description, wer, cer))

    if print_table: 
        print(tabulate(table, headers=["Source & Params", "WER", "CER"], tablefmt="grid"))

    return table
