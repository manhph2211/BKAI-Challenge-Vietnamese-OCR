import os
import cv2 
import json 
import numpy as np


def get_datalist(train_json_path="../data/train_data.json", test_json_path="../data/test_data.json"):
    data_list = []
    with open(train_json_path, "r") as F:
        train_list = json.load(F)
        data_list.extend(train_list)
    with open(test_json_path, "r") as f:
        test_list = json.load(f)
        data_list.extend(test_list)
    
    return data_list


def parse_text(gt_path):
    polygons = []
    with open(gt_path, "r") as F:
        contents = F.readlines()
    for line in contents:
        poly = []
        content = line.split(",")
        poly.extend([int(i) for i in content[:8]])
        text_label = line.replace(",".join(content[:8]) + ",", "")
        
        xx = [poly[2*i] for i in range(4)]
        yy = [poly[2*i+1] for i in range(4)]
        pts = np.stack([xx, yy]).T.astype(np.int32)

        d1 = np.linalg.norm(pts[0] - pts[1])
        d2 = np.linalg.norm(pts[1] - pts[2])
        d3 = np.linalg.norm(pts[2] - pts[3])
        d4 = np.linalg.norm(pts[3] - pts[0])

        if min([d1, d2, d3, d4]) < 4: 
            continue
        polygons.append({"poly": poly, "text_label": text_label})
    
    return polygons


def get_background(image, polygons):
    mask = np.zeros(image.shape[:2], dtype="uint8")
    for polygon in polygons:
        poly = polygon["poly"]
        text_label = polygon["text_label"]

        xx = [poly[2 * i] for i in range(4)]
        yy = [poly[2 * i + 1] for i in range(4)]

        points = np.array([(xx[i], yy[i]) for i in range(4)]).astype(np.int32)

        cv2.fillPoly(mask, pts=[points], color=1)

    # Uncomment to show the mask
    # plt.imshow(np.expand_dims(mask, 2)*image)
    # plt.show()

    background_image = cv2.inpaint(image, mask, 3, cv2.INPAINT_NS)

    return background_image


def create_background(data_list, dest_folder):
    for idx, data in enumerate(data_list):
        image = cv2.imread('../' + data["image_path"])
        polygons = parse_text('../' + data["gt_path"])
        bg_image = get_background(image, polygons)
        bg_path = os.path.join(dest_folder, data["image_path"].split("/")[-1])
        cv2.imwrite(bg_path, bg_image)
  
    return

if __name__ == "__main__":
    data_list = get_datalist()
    create_background(data_list, dest_folder="./trdg/images")

