import sys
import os
import  glob
import lmdb # install lmdb by "pip install lmdb"
import cv2
import yaml
from yaml.loader import SafeLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt 
from PIL import Image
from configs import Cfg
from src.utils.predictor import Predictor


def checkImageIsValid(imageBin):
    isvalid = True
    imgH = None
    imgW = None

    imageBuf = np.fromstring(imageBin, dtype=np.uint8)
    try:
        img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)

        imgH, imgW = img.shape[0], img.shape[1]
        if imgH * imgW == 0:
            isvalid = False
    except Exception as e:
        isvalid = False

    return isvalid, imgH, imgW


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k.encode(), v)


def createDataset(outputPath, root_dir, annotation_path):
    """
    Create LMDB dataset for CRNN training.
    ARGS:
        outputPath    : LMDB output path
        imagePathList : list of image path
        labelList     : list of corresponding groundtruth texts
        lexiconList   : (optional) list of lexicon lists
        checkValid    : if true, check the validity of every image
    """

    annotation_path = os.path.join(root_dir, annotation_path)
    with open(annotation_path, 'r') as ann_file:
        lines = ann_file.readlines()
        annotations = [l.strip().split('\t') for l in lines]

    nSamples = len(annotations)
    env = lmdb.open(outputPath, map_size=1099511627776)
    cache = {}
    cnt = 0
    error = 0
    
    pbar = tqdm(range(nSamples), ncols = 100, desc='Create {}'.format(outputPath)) 
    for i in pbar:
        imageFile, label = annotations[i]
        imagePath = os.path.join(root_dir, imageFile)

        if not os.path.exists(imagePath):
            error += 1
            continue
        
        with open(imagePath, 'rb') as f:
            imageBin = f.read()
        isvalid, imgH, imgW = checkImageIsValid(imageBin)

        if not isvalid:
            error += 1
            continue

        imageKey = 'image-%09d' % cnt
        labelKey = 'label-%09d' % cnt
        pathKey = 'path-%09d' % cnt
        dimKey = 'dim-%09d' % cnt

        cache[imageKey] = imageBin
        cache[labelKey] = label.encode()
        cache[pathKey] = imageFile.encode()
        cache[dimKey] = np.array([imgH, imgW], dtype=np.int32).tobytes()

        cnt += 1

        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}

    nSamples = cnt-1
    cache['num-samples'] = str(nSamples).encode()
    writeCache(env, cache)

    if error > 0:
        print('Remove {} invalid images'.format(error))
    print('Created dataset with %d samples' % nSamples)
    sys.stdout.flush()


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
            crop_img = cut_img(img,offset)
            crop_img.save(os.path.join(img_path.replace(".jpg",""),f'{idx}.jpg'))
            crop_img = Image.open(os.path.join(img_path.replace(".jpg",""),f'{idx}.jpg'))
            s = detector.predict(crop_img)
            line[-1] = str(s)+'\n' if idx+1<len(lines) else str(s)
            line = ','.join(line)
            save_line += line
    with open(os.path.join("/home/edabk/phumanhducanh/BKAI/prediction",txt_path.split('/')[-1]), 'w') as f:
            f.write(save_line)


def cut_img(img,offsets):
    x1,y1,x2,y2,x3,y3,x4,y4 = offsets
    top_left_x = min([x1,x2,x3,x4])
    top_left_y = min([y1,y2,y3,y4])
    bot_right_x = max([x1,x2,x3,x4])
    bot_right_y = max([y1,y2,y3,y4])
    crop_img = img[top_left_y:bot_right_y+1, top_left_x:bot_right_x+1]
    return Image.fromarray(crop_img)
    # plt.imshow(crop_img)
    # plt.show()


def get_info_from_img_name(img_path, config):
    total = []
    label_path = os.path.join(config['dataset']['data_root'],config['dataset']['labels_folder'],img_path.split('/')[-1].replace('jpg','txt'))
    with open(label_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.split(',')
            text = line[-1][:-1]
            if len(text) == 0:
                continue
            offsets = line[:-1]
            total.append([offsets,text])
    return total


def create_anno(config):
    save_path = os.path.join(config['dataset']['data_root'],config['dataset']['save_folder'])
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
        os.mkdir(os.path.join(save_path,'all_imgs'))
    save_all_img_folder = os.path.join(save_path,'all_imgs')
    train_img_paths = glob.glob(os.path.join(config['dataset']['data_root'],config['dataset']['train_folder'],'*.jpg'))
    test_img_paths = glob.glob(os.path.join(config['dataset']['data_root'],config['dataset']['valid_folder'],'*.jpg'))
    train_annotation = os.path.join(save_path,config['dataset']['train_annotation'])
    test_annotation = os.path.join(save_path,config['dataset']['valid_annotation'])

    idx = 1
    with open(train_annotation, 'w') as f:
        for img_path in tqdm(train_img_paths):
            img = np.array(Image.open(img_path))
            for offsets, text in get_info_from_img_name(img_path,config):
                try:
                    if "##" in text:
                        continue
                    crop_img = cut_img(img, (int(offsets[0]),int(offsets[1]),int(offsets[2]),int(offsets[3]),int(offsets[4]),int(offsets[5]),int(offsets[6]),int(offsets[7])))
                    crop_img.save(os.path.join(save_all_img_folder,f'{idx}.jpg'))
                    f.write('all_imgs/' + f'{idx}.jpg\t' + text + '\n')
                    
                    idx += 1
                except:
                    print("Label might be wrong !")

    with open(test_annotation, 'w') as f:
        for img_path in tqdm(test_img_paths):
            img = np.array(Image.open(img_path))
            for offsets, text in get_info_from_img_name(img_path,config):
                try:
                    if "##" in text:
                        continue
                    crop_img = cut_img(img, (int(offsets[0]),int(offsets[1]),int(offsets[2]),int(offsets[3]),int(offsets[4]),int(offsets[5]),int(offsets[6]),int(offsets[7])))
                    crop_img.save(os.path.join(save_all_img_folder, f'{idx}.jpg'))
                    f.write('all_imgs/' + f'{idx}.jpg\t' + text + '\n')
                    idx += 1
                except:
                    print("Label might be wrong !")


if __name__ == '__main__':
    #with open('/home/edabk/phumanhducanh/BKAI/TransOCR-Pytorch/configs/base.yml', 'r') as f:
    #    config = yaml.load(f, Loader=SafeLoader)
    #create_anno(config)
    config= "/home/edabk/phumanhducanh/BKAI/TransOCR-Pytorch/configs/vgg-transformer.yml"
    demo_dir = "/home/edabk/phumanhducanh/BKAI/DB/demo_results"
    txt_paths = glob.glob(os.path.join(demo_dir,"*.txt"))
    
    config = Cfg.load_config_from_file(config)
    detector = Predictor(config)
    for txt_path in tqdm(txt_paths):
        run(detector, txt_path)

    
