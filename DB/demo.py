#!python3
import argparse
import os
import torch
import cv2
import numpy as np
from experiment import Structure, Experiment
from concern.config import Configurable, Config
import math
import glob 
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description='Text Recognition Training')
    parser.add_argument('exp', type=str)
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    parser.add_argument('--image_path', type=str, help='image path')
    parser.add_argument('--result_dir', type=str, default='./demo_results/', help='path to save results')
    parser.add_argument('--data', type=str,
                        help='The name of dataloader which will be evaluated on.')
    parser.add_argument('--image_short_side', type=int, default=736,
                        help='The threshold to replace it in the representers')
    parser.add_argument('--thresh', type=float,
                        help='The threshold to replace it in the representers')
    parser.add_argument('--box_thresh', type=float, default=0.6,
                        help='The threshold to replace it in the representers')
    parser.add_argument('--visualize', action='store_true',
                        help='visualize maps in tensorboard')
    parser.add_argument('--resize', action='store_true',
                        help='resize')
    parser.add_argument('--polygon', action='store_true',
                        help='output polygons if true')
    parser.add_argument('--margin_h', type=int, default=0, 
                        help='remove margin in an annotation file')
    parser.add_argument('--margin_w', type=int, default=0, 
                        help='remove margin in an annotation file')
    parser.add_argument('--margin_scale_w', type=float, default=0.0,
                        help='remove margin in an annotation file')
    parser.add_argument('--margin_scale_h', type=float, default=0.0,
                        help='remove margin in an annotation file')
    parser.add_argument('--eager', '--eager_show', action='store_true', dest='eager_show',
                        help='Show iamges eagerly')

    args = parser.parse_args()
    args = vars(args)
    args = {k: v for k, v in args.items() if v is not None}

    conf = Config()
    experiment_args = conf.compile(conf.load(args['exp']))['Experiment']
    experiment_args.update(cmd=args)
    experiment = Configurable.construct_class_from_config(experiment_args)
    print("Resuming from " + args['resume'])
    pad_folder = '/'.join(args['image_path'].split('/')[:-1]) + '/pad_img/'
    print(pad_folder)
    if not os.path.isdir(pad_folder):
        os.mkdir(pad_folder)
    if os.path.isdir(args['image_path']):
        for img_path in tqdm(glob.glob(os.path.join(args['image_path'],"*.jpg"))):
            if img_path.endswith('jpg'):
                image = cv2.imread(img_path, cv2.IMREAD_COLOR)
  
                if args['margin_h'] != 0 or args['margin_w'] != 0:  
                    new_image = np.zeros((image.shape[0] + 2 * args['margin_h'], image.shape[1] + 2 * args['margin_w'], image.shape[2]))
                    new_image[args['margin_h']: image.shape[0] + args['margin_h'], args['margin_w']: image.shape[1] + args['margin_w'], :] = image

                if args['margin_scale_h'] != 0.0 or args['margin_scale_w'] != 0.0:
                    new_image = np.zeros((image.shape[0] + 2 * int(image.shape[0] * args['margin_scale_h']), image.shape[1] + 2 * int(image.shape[1] * args['margin_scale_w']), image.shape[2]))
                    new_image[int(image.shape[0] * args['margin_scale_h']): image.shape[0] + int(image.shape[0] * args['margin_scale_h']), int(image.shape[1] * args['margin_scale_w']): image.shape[1] + int(image.shape[1] * args['margin_scale_w']), :] = image

                save_path = '/'.join(img_path.split('/')[:-2]) + '/pad_img/' + img_path.split('/')[-1]
                cv2.imwrite(save_path, new_image)
                Demo(experiment, experiment_args, cmd=args).inference(save_path, args['visualize'])
    else:
        Demo(experiment, experiment_args, cmd=args).inference(args['image_path'], args['visualize'])



class Demo:
    def __init__(self, experiment, args, cmd=dict()):
        self.RGB_MEAN = np.array([122.67891434, 116.66876762, 104.00698793])
        self.experiment = experiment
        experiment.load('evaluation', **args)
        self.args = cmd
        model_saver = experiment.train.model_saver
        self.structure = experiment.structure
        self.model_path = self.args['resume']

    def init_torch_tensor(self):
        # Use gpu or not
        torch.set_default_tensor_type('torch.FloatTensor')
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            self.device = torch.device('cpu')

    def init_model(self):
        model = self.structure.builder.build(self.device)
        return model

    def resume(self, model, path):
        if not os.path.exists(path):
            print("Checkpoint not found: " + path)
            return
        # print("Resuming from " + path)
        states = torch.load(
            path, map_location=self.device)
        model.load_state_dict(states, strict=False)
        # print("Resumed from " + path)

    def resize_image(self, img):
        height, width, _ = img.shape
        if height < width:
            new_height = self.args['image_short_side']
            new_width = int(math.ceil(new_height / height * width / 32) * 32)
        else:
            new_width = self.args['image_short_side']
            new_height = int(math.ceil(new_width / width * height / 32) * 32)
        resized_img = cv2.resize(img, (new_width, new_height))
        return resized_img
        
    def load_image(self, image_path):
        img = cv2.imread(image_path, cv2.IMREAD_COLOR).astype('float32')
        original_shape = img.shape[:2]
        img = self.resize_image(img)
        img -= self.RGB_MEAN
        img /= 255.
        img = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0)
        return img, original_shape
        
    def format_output(self, batch, output):
        batch_boxes, batch_scores = output
        for index in range(batch['image'].size(0)):
            original_shape = batch['shape'][index]
            filename = batch['filename'][index]
            result_file_name = 'res_' + filename.split('/')[-1].split('.')[0] + '.txt'
            result_file_path = os.path.join(self.args['result_dir'], result_file_name)
            boxes = batch_boxes[index]
            scores = batch_scores[index]
            if self.args['polygon']:
                with open(result_file_path, 'wt') as res:
                    for i, box in enumerate(boxes):
                        box = np.array(box).reshape(-1).tolist()
                        result = ",".join([str(int(x)) for x in box])
                        score = scores[i]
                        res.write(result + ',' + str(score) + "\n")
            else:
                with open(result_file_path, 'wt') as res:
                    for i in range(boxes.shape[0]):
                        score = scores[i]
                        if score < self.args['box_thresh']:
                            continue
                        box = boxes[i,:,:].reshape(-1).tolist()
                        result = ",".join([str(int(x)) for x in box])
                        res.write(result + ',' + str(score) + "\n")
        
    def inference(self, image_path, visualize=False):
        self.init_torch_tensor()
        model = self.init_model()
        self.resume(model, self.model_path)
        all_matircs = {}
        model.eval()
        batch = dict()
        batch['filename'] = [image_path]
        img, original_shape = self.load_image(image_path)

        batch['shape'] = [original_shape]
        with torch.no_grad():
            batch['image'] = img
            pred = model.forward(batch, training=False)
            output = self.structure.representer.represent(batch, pred, is_output_polygon=self.args['polygon']) 
            if not os.path.isdir(self.args['result_dir']):
                os.mkdir(self.args['result_dir'])
            self.format_output(batch, output)

            if visualize and self.structure.visualizer:
                vis_image = self.structure.visualizer.demo_visualize(image_path, output)
                cv2.imwrite(os.path.join(self.args['result_dir'], image_path.split('/')[-1].split('.')[0]+'.jpg'), vis_image)

            predict_path = os.path.join(self.args['result_dir'], 'res_' + image_path.split('/')[-1].split('.')[0]+'.jpg').replace('jpg', 'txt')
            new_lines = ''

            with open(predict_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    coor = [int(i) for i in line.split(',')[:8]]
                    for i in range(4):
						# Constant padding
                        coor[2 * i + 1] = min(vis_image.shape[0] - 2 * self.args['margin_h'], max(0, coor[2 * i + 1] - self.args['margin_h']))
                        coor[2 * i] = min(vis_image.shape[1] - 2 * self.args['margin_w'], max(0, coor[2 * i] - self.args['margin_w']))


						# Scale padding
                        pad_size = [int(float(vis_image.shape[0]) / (1 + 2 * self.args['margin_scale_h']) * self.args[
                            'margin_scale_h']),
                                    int(float(vis_image.shape[1]) / (1 + 2 * self.args['margin_scale_w']) * self.args[
                                        'margin_scale_w'])]

                        coor[2 * i + 1] = min(vis_image.shape[0] - 2 * pad_size[0],
                                              max(0, coor[2 * i + 1] - pad_size[0]))
                        coor[2 * i] = min(vis_image.shape[1] - 2 * pad_size[1], max(0, coor[2 * i] - pad_size[1]))

                    coor = [str(i) for i in coor]
                    coor = ','.join(coor)
                    coor += ',' + ','.join(line.split(',')[8:])
                    new_lines += coor
            with open(predict_path, 'w') as f:
                f.write(new_lines)

if __name__ == '__main__':
    main()
