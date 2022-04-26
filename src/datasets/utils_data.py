import os
from glob import glob
import json 

import random
from types import new_class
import matplotlib.pyplot as plt
import numpy as np
import cv2


def draw(image, anns):
  for item in anns:
    poly = item["poly"]
    text = item["text"]
    points = [(poly[2*i], poly[2*i+1]) for i in range(0,4)]
    color = (0, 255, 0)
    thickness = 2
    for i in range(0, 4):
      cv2.line(image, points[i], points[(i+1)%4], color, thickness)
  plt.figure(figsize = (10, 10))
  plt.imshow(image)
  plt.show()

  return

  
def export_data(data_folder="../../data"):
  data = []

  # Training data from BKAI
  BKAI_img_folder = os.path.join(data_folder, "training_img/")
  BKAI_gt_folder = os.path.join(data_folder, "training_gt/")

  for gt_file in os.listdir(BKAI_gt_folder):
    image_file = gt_file[3:-4] + ".jpg" 
    item = {}
    item["image_path"] = os.path.join(BKAI_img_folder, image_file)
    item["gt_path"] = os.path.join(BKAI_gt_folder, gt_file)
    data.append(item)

  # Training data from VINAI
  VINAI_gt_folder = os.path.join(data_folder, "vietnamese/labels/")
  list_images = glob(os.path.join(data_folder, "vietnamese") + "/*/*.jpg")

  for image_file in list_images:
    image_idx = int(image_file.split("/")[-1][2:-4])
    gt_file = "".join(["gt_", str(image_idx), ".txt"]) 
    item = {}
    item["image_path"] = image_file
    item["gt_path"] = os.path.join(VINAI_gt_folder, gt_file)
    data.append(item)

  return data


def load_ann(gt_path):
  items = []
  with open(gt_path, "r") as F:
    contents = F.readlines()
    for line in contents:
      item = {}
      poly = []
      content = line.split(",")
      poly.extend([int(i) for i in content[:8]])
      text_label = line.replace(",".join(content[:8]) + ",", "") 
      item["poly"] = poly
      item["text"] = text_label
      items.append(item)
  return items


def save_json(data, json_file_name):
  with open(json_file_name, "w") as f:
    json.dump(data, f, indent=4)
  return 


def load_json(json_file_name):
  with open(json_file_name, "r") as f:
    data = json.load(f)
  return data


def rename(json_file_name):
    data = load_json(json_file_name)
    for i,item in enumerate(data):
        new_image_path = item["image_path"].replace(item["image_path"].split('/')[-1],str(i)+'.jpg')
        os.rename(item["image_path"],new_image_path)

        new_gt_path = item["gt_path"].replace(item["gt_path"].split('/')[-1],str(i)+'.txt')
        os.rename(item["gt_path"],new_gt_path)

        item["image_path"] = new_image_path
        item["gt_path"] = new_gt_path

    save_json(data, json_file_name)


if __name__ == "__main__":
    data = export_data('/home/edabk/phumanhducanh/BKAI/data')
    json_file_name = '/home/edabk/phumanhducanh/BKAI/data/vietnamese/data.json'
    save_json(data,json_file_name)
    random_idx = random.randrange(0, len(data)-1)
    print(data[random_idx]["image_path"])
    print("Index:", random_idx)
    #anns = load_ann(data[random_idx]["gt_path"])
    #image_test = cv2.imread(data[random_idx]["image_path"])
    # draw(image_test, anns)
    rename(json_file_name)
