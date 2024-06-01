import doxapy
import cv2
import os
import shutil
from utils import *

img_dir = "./imgs/pt"
save_dir = "./texts/ocr_from_col"

extract_text(img_dir, save_dir)
mean_wer, mean_cer, word_error_rate, char_error_rate = evaluate("./texts/gt", save_dir, print_results=False)
res_table = put_res_to_table("Colored cropped images", mean_wer, mean_cer, False)

# Try to binarize the images on your own (e.g. Otsu, Su) and compare it to the OCR results of the color images

su = doxapy.Binarization(doxapy.Binarization.Algorithms.SU)

bin_imgs_dir = "./imgs/pt_bin"

if not os.path.exists(bin_imgs_dir):
    os.makedirs(bin_imgs_dir)
else:
    shutil.rmtree(bin_imgs_dir)
    os.makedirs(bin_imgs_dir)

for i in os.listdir(img_dir):
    img_idx = i.split(".")[0].split("_")[1]
    img = cv2.imread(img_dir + "/" + i, cv2.IMREAD_GRAYSCALE)
    su.initialize(img)
    su.to_binary(img, {"window": 10})
    cv2.imwrite(bin_imgs_dir + "/bin_" + img_idx + ".jpg", img)  

extract_text(bin_imgs_dir, "./texts/ocr_from_bin")
mean_wer, mean_cer, word_error_rate, char_error_rate = evaluate("./texts/gt", "./texts/ocr_from_bin", print_results=False)
res_table = put_res_to_table("Binarized cropped images (Su, window size=15)", mean_wer, mean_cer, False, res_table)

# Directly forward the full images to Tesseract and compare it to the results when only using the crops of our Yolo model

full_img_dir = "./imgs/dataset"
full_img_copy_dir = "./imgs/full_img_copy"

if not os.path.exists(full_img_copy_dir):
    os.makedirs(full_img_copy_dir)
else:
    shutil.rmtree(full_img_copy_dir)
    os.makedirs(full_img_copy_dir)   

for i in os.listdir(full_img_dir):
    img_idx = str(int(i.split("_")[0]))
    shutil.copyfile(full_img_dir + "/" + i, full_img_copy_dir + "/img_" + img_idx + ".jpg")

extract_text(full_img_copy_dir, "./texts/ocr_from_full")
mean_wer, mean_cer, word_error_rate, char_error_rate = evaluate("./texts/gt", "./texts/ocr_from_full", print_results=False)
res_table = put_res_to_table("Directly forwarded full images (psm=3)", mean_wer, mean_cer, False, res_table)

# Evaluate the influence of the Page Segmentation Mode (PSM) of Tesseract

extract_text(full_img_copy_dir, "./texts/bin_psm_1", config="--psm 1")
mean_wer, mean_cer, word_error_rate, char_error_rate = evaluate("./texts/gt", "./texts/bin_psm_1", print_results=False)
res_table = put_res_to_table("Directly forwarded full images (psm=1)", mean_wer, mean_cer, False, res_table)

extract_text(full_img_copy_dir, "./texts/bin_psm_11", config="--psm 11")
mean_wer, mean_cer, word_error_rate, char_error_rate = evaluate("./texts/gt", "./texts/bin_psm_11", print_results=False)
res_table = put_res_to_table("Directly forwarded full images (psm=11)", mean_wer, mean_cer, False, res_table)

extract_text(bin_imgs_dir, "./texts/bin_psm6", config="--psm 6")
mean_wer, mean_cer, word_error_rate, char_error_rate = evaluate("./texts/gt", "./texts/bin_psm6", print_results=False)
res_table = put_res_to_table("Binarized cropped images (Su, window size=15, psm=6)", mean_wer, mean_cer, False, res_table)

extract_text(img_dir, "./texts/col_psm_6", config="--psm 6")
mean_wer, mean_cer, word_error_rate, char_error_rate = evaluate("./texts/gt", "./texts/col_psm_6", print_results=False)
res_table = put_res_to_table("Colored cropped images (psm=6)", mean_wer, mean_cer, False, res_table)

extract_text(bin_imgs_dir, "./texts/bin_psm_11", config="--psm 11")
mean_wer, mean_cer, word_error_rate, char_error_rate = evaluate("./texts/gt", "./texts/bin_psm_11", print_results=False)
res_table = put_res_to_table("Binarized cropped images (Su, window size=15, psm=11)", mean_wer, mean_cer, True, res_table)