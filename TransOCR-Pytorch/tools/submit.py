import sys
import os
import  glob
import lmdb # install lmdb by "pip install lmdb"
import cv2
import yaml
import math
from yaml.loader import SafeLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
from configs import Cfg
from src.utils.predictor import Predictor
from src.utils.create_dataset import crop_nonrectangle


def run(detector, txt_path,save_folder="/home/edabk/phumanhducanh/BKAI/prediction",data_folder='/home/edabk/phumanhducanh/BKAI/data/public_test_img'):
    img_path = os.path.join(data_folder, txt_path.split('/')[-1].replace('res_', '').replace('txt', 'jpg'))
    if not os.path.isdir(img_path.replace(".jpg","")):
        os.makedirs(img_path.replace(".jpg",""))
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)

    with open(txt_path,"r") as f:
        lines = f.readlines()
        if len(lines) == 0:
            lines = [f"0,0,{img.shape[1]},0,{img.shape[1]},{img.shape[0]},0,{img.shape[0]},0"]
            print(txt_path)
        save_line = ''
        for idx,line in enumerate(lines):
            line = line.split(",")
            assert len(line) == 9
            offset = [int(num) for num in line[:-1]]
            crop_img = crop_nonrectangle(img, offset)
            #crop_img.save(os.path.join(img_path.replace(".jpg",""),f'{idx}.jpg'))
            cv2.imwrite(os.path.join(img_path.replace(".jpg", ""),f'{idx}.jpg'), crop_img)
            crop_img = Image.open(os.path.join(img_path.replace(".jpg",""),f'{idx}.jpg'))
            s = detector.predict(crop_img)
            line[-1] = str(s)+'\n' if idx+1<len(lines) else str(s)
            line = ','.join(line)
            save_line += line
    with open(os.path.join("/home/edabk/phumanhducanh/BKAI/prediction",txt_path.split('/')[-1]), 'w') as f:
            f.write(save_line)


if __name__ == '__main__':
    # with open('/home/edabk/phumanhducanh/BKAI/TransOCR-Pytorch/configs/base.yml', 'r') as f:
    #    config = yaml.load(f, Loader=SafeLoader)
    # create_anno(config)
    config = "/home/edabk/phumanhducanh/BKAI/TransOCR-Pytorch/configs/vgg-transformer.yml"
    demo_dir = "/home/edabk/phumanhducanh/BKAI/DB/demo_results"
    txt_paths = glob.glob(os.path.join(demo_dir, "*.txt"))

    config = Cfg.load_config_from_file(config)
    detector = Predictor(config)

    for txt_path in tqdm(txt_paths):
        run(detector, txt_path)
