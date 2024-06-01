import os
import shutil
import warnings
warnings.filterwarnings("ignore")
from ultralytics import YOLO

warnings.filterwarnings("ignore")

model = YOLO("./yolov8n.pt")

dt_path = "./imgs/dataset"
imgs_paths = []

for i in os.listdir(dt_path):
    if i.endswith(".jpg"):
        imgs_paths.append(os.path.join(dt_path, i))

imgs_paths.sort()

results = model(imgs_paths, classes=[2], verbose=False)

save_dir = "./imgs/results"

if os.path.exists(save_dir):
    shutil.rmtree(save_dir)
    os.makedirs(save_dir)

for idx, res in enumerate(results):
    res.save_crop(save_dir + "/img_" + str(idx + 1))

pt_dir = "./imgs/pt"

if not os.path.exists(pt_dir):
    os.makedirs(pt_dir)
else:
    shutil.rmtree(pt_dir, ignore_errors=True)
    os.makedirs(pt_dir)

for res in os.listdir(save_dir):
    old_name = os.path.join(save_dir, res, "pt", "im.jpg")
    new_name = os.path.join(save_dir, res, "pt", "img_" + res.split("_")[1] + ".jpg")

    os.rename(old_name, new_name)
    shutil.copy(new_name, pt_dir)

shutil.rmtree(save_dir, ignore_errors=True)
