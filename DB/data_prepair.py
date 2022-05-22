import os
import json
from sklearn.model_selection import train_test_split
from glob import glob


def export_data(data_folder="datasets"):
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
    image_idx = int(image_file.split("/")[-1][:-4])
    gt_file = "".join([str(image_idx), ".txt"]) 
    item = {}
    item["image_path"] = image_file
    item["gt_path"] = os.path.join(VINAI_gt_folder, gt_file)
    data.append(item)

  return data


def export_test_data(data_folder='public_img_test'):
    data = []

    for image_name in os.listdir(data_folder):
        image_path = os.path.join(data_folder, image_name)

        data.append(image_path)

    return data


if __name__ == '__main__':
    data = export_data('datasets')

    train_data, test_data = train_test_split(data, test_size=0.001)

    print(f'Train size: {len(train_data)}')
    print(f'Test size: {len(test_data)}') 

    with open('datasets/train_data_db.json', 'w') as f:
        json.dump(train_data, f)

    with open('datasets/test_data_db.json', 'w') as f:
        json.dump(test_data, f)    

