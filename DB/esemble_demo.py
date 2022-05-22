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
    # Arguments Yaml files
    parser.add_argument('--exp_db', default=None, type=str, help='Yaml file config for DBnet model')
    parser.add_argument('--exp_dbplus', default=None, type=str, help='Yaml file config for DBnet++ model')

    # Arguments checkpoints
    parser.add_argument('--checkpoint_db', default=None, type=str, help='checkpoint for DBnet model')
    parser.add_argument('--checkpoint_dbplus',default=None, type=str, help='checkpoint for DBnet++ model')

    # Input and hyper params
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
                        help='Show images eagerly')

    args = parser.parse_args()
    args = vars(args)
    args = {k: v for k, v in args.items() if v is not None}

    # Create configs
    if args['checkpoint_db'] is not None and args['exp_db'] is not None:
        db_conf = Config()
        db_experiment_args = db_conf.compile(db_conf.load(args['exp_db']))['Experiment']
        db_experiment_args.update(cmd=args)
        db_experiment = Configurable.construct_class_from_config(db_experiment_args)
        print("Resuming DBnet from " + args['checkpoint_db'])
    if args['checkpoint_dbplus'] is not None and args["exp_dbplus"] is not None:
        dbplus_conf = Config()
        dbplus_experiment_args = dbplus_conf.compile(dbplus_conf.load(args['exp_db']))['Experiment']
        dbplus_experiment_args.update(cmd=args)
        dbplus_experiment = Configurable.construct_class_from_config(dbplus_experiment_args)
        print("Resuming DBnet++ from " + args['checkpoint_dbplus'])
    else:
        raise("You should give checkpoint and exp of the same model !!!")

    # conf = Config()
    # experiment_args = conf.compile(conf.load(args['exp']))['Experiment']
    # experiment_args.update(cmd=args)
    # experiment = Configurable.construct_class_from_config(experiment_args)
    # print("Resuming from " + args['resume'])

    if os.path.isdir(args['image_path']):
        for img_path in tqdm(glob.glob(os.path.join(args['image_path'], "*.jpg"))):
            if img_path.endswith('jpg'):
                image = cv2.imread(img_path, cv2.IMREAD_COLOR)
                #new_image = np.zeros((image.shape[0] + 2 * args['margin'], image.shape[1], image.shape[2]))
                #new_image[args['margin']: image.shape[0] + args['margin'], :, :] = image
                if (args['margin_h'] != 0) or (args['margin_w'] != 0): 
                    new_image = np.zeros((image.shape[0] + 2 * args['margin_h'], image.shape[1] + 2 * args['margin_w'], image.shape[2]))
                    new_image[args['margin_h']: image.shape[0] + args['margin_h'], args['margin_w']: image.shape[1] + args['margin_w'], :] = image

                if (args['margin_scale_h'] != 0.0) or (args['margin_scale_w'] != 0.0):
                    new_image = np.zeros((image.shape[0] + 2 * int(image.shape[0] * args['margin_scale_h']), image.shape[1] + 2 * int(image.shape[1] * args['margin_scale_w']), image.shape[2]))
                    new_image[int(image.shape[0] * args['margin_scale_h']): image.shape[0] + int(image.shape[0] * args['margin_scale_h']), int(image.shape[1] * args['margin_scale_w']): image.shape[1] + int(image.shape[1] * args['margin_scale_w']), :] = image


                save_path = '/'.join(img_path.split('/')[:-2]) + '/pad_img/' + img_path.split('/')[-1]
                cv2.imwrite(save_path, new_image)
                Demo(db_experiment, db_experiment_args,dbplus_experiment, dbplus_experiment_args, cmd=args).inference(save_path, args['visualize'])
    else:
        Demo(db_experiment, db_experiment_args, dbplus_experiment, dbplus_experiment_args, cmd=args).inference(args['image_path'], args['visualize'])


class Demo:
    def __init__(self, db_experiment, db_args, dbplus_experiment, dbplus_args, cmd=dict()):
        self.RGB_MEAN = np.array([122.67891434, 116.66876762, 104.00698793])
        self.args = cmd

        # Create structure and model path for DBnet
        self.db_experiment = db_experiment
        db_experiment.load('evaluation', **db_args)

        model_saver = db_experiment.train.model_saver
        self.db_structure = db_experiment.structure
        self.db_model_path = self.args['checkpoint_db']

        # Create structure and model path for DBnet++
        self.dbplus_experiment = dbplus_experiment
        dbplus_experiment.load('evaluation', **dbplus_args)
        model_saver = dbplus_experiment.train.model_saver
        self.dbplus_structure = dbplus_experiment.structure
        self.dbplus_model_path = self.args['checkpoint_dbplus']

    def init_torch_tensor(self):
        # Use gpu or not
        torch.set_default_tensor_type('torch.FloatTensor')
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            self.device = torch.device('cpu')

    def init_model(self):
        db_model = self.db_structure.builder.build(self.device)
        dbplus_model = self.dbplus_structure.builder.build(self.device)

        return db_model, dbplus_model

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

    def format_output(self, batch, db_output, dbplus_output):
        db_batch_boxes, db_batch_scores = db_output
        dbplus_batch_boxes, dbplus_batch_scores = dbplus_output
        batch_boxes = []
        batch_scores = []

        for index in range(batch['image'].size(0)):
            original_shape = batch['shape'][index]
            filename = batch['filename'][index]
            result_file_name = 'res_' + filename.split('/')[-1].split('.')[0] + '.txt'
            result_file_path = os.path.join(self.args['result_dir'], result_file_name)

            # merge the db_output and bplus_output
            boxes = np.concatenate([db_batch_boxes[index], dbplus_batch_boxes[index]], axis=0)
            scores = np.concatenate([db_batch_scores[index], dbplus_batch_scores[index]], axis=0)

            # Apply nms
            boxes, scores = self.remove_overlapping_boxes(boxes, scores)
            boxes, scores = self.nms(boxes, scores)
            #boxes, scores = self.remove_overlapping_boxes(boxes, scores)
            batch_boxes.append(boxes)
            batch_scores.append(scores)

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
                        box = boxes[i, :, :].reshape(-1).tolist()
                        result = ",".join([str(int(x)) for x in box])
                        res.write(result + ',' + str(score) + "\n")
        return batch_boxes, batch_scores

    def nms(self, boxes, scores, thresh=0.6):
        assert boxes.shape[0] == scores.shape[0]
        x1 = boxes[:, 0, 0]
        y1 = boxes[:, 0, 1]
        x2 = boxes[:, 2, 0]
        y2 = boxes[:, 2, 1]

        x3 = boxes[:, 1, 0]
        y3 = boxes[:, 1, 1]
        x4 = boxes[:, 3, 0]
        y4 = boxes[:, 3, 1]

        xmin = np.min(np.stack([x1, x2, x3, x4], axis=1), axis=1).astype(float)
        ymin = np.min(np.stack([y1, y2, y3, y4], axis=1), axis=1).astype(float) 
        xmax = np.max(np.stack([x1, x2, x3, x4], axis=1), axis=1).astype(float)
        ymax = np.max(np.stack([y1, y2, y3, y4], axis=1), axis=1).astype(float)
	
        assert len(xmin.shape) == 1, "Wrong shape x1: {}".format(x1.shape)
        areas = (xmax - xmin + 1) * (ymax - ymin + 1)
        areas = areas.astype(float)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            order = order[1:]
            xx1 = np.maximum(xmin[i], xmin[order])
            yy1 = np.maximum(ymin[i], ymin[order])
            xx2 = np.minimum(xmax[i], xmax[order])
            yy2 = np.minimum(ymax[i], ymax[order])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order] - inter)

            inds = np.where(ovr <= thresh)[0]
            order = order[inds]

        nms_boxes = boxes[keep, :, :].copy()
        nms_scores = scores[keep].copy()
        return nms_boxes, nms_scores


    def remove_overlapping_boxes(self, boxes, scores, thresh=0.6):
        """keep = {i: 1 for i in boxes.shape[0]}
        x1 = boxes[:, 0, 0]
        y1 = boxes[:, 0, 1]
        x2 = boxes[:, 2, 0]
        y2 = boxes[:, 2, 1]
        assert len(x1.shape) == 1, "Wrong shape x1: {}".format(x1.shape)
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)

		for id1 in range(boxes.shape[0]):
		    for id2 in range(len(boxes.shape[0]) and id2 != id1:
			    
                xx1 = np.maximum(boxes[id1][0][0], boxes[id2][0][0])
                yy1 = np.maximum(boxes[id1][0][1], boxes[id2][0][1])
                xx2 = np.minimum(boxes[id1][2][0], boxes[id2][2][0])
                yy2 = np.minimum(boxes[id1][2][1], boxes[id2][2][1])

                w = np.maximum(0.0, xx2 - xx1 + 1)
                h = np.maximum(0.0, yy2 - yy1 + 1)
                inter = w * h
			
                ovr = inter / min(areas[id1], areas[id2])

				if ovr > thresh:
                    if areas[id1] > are"""
                    
        x1 = boxes[:, 0, 0]
        y1 = boxes[:, 0, 1]
        x2 = boxes[:, 2, 0]
        y2 = boxes[:, 2, 1]

        x3 = boxes[:, 1, 0]
        y3 = boxes[:, 1, 1]
        x4 = boxes[:, 3, 0]
        y4 = boxes[:, 3, 1]

        xmin = np.min(np.stack([x1, x2, x3, x4], axis=1), axis=1).astype(float)
        ymin = np.min(np.stack([y1, y2, y3, y4], axis=1), axis=1).astype(float)
        xmax = np.max(np.stack([x1, x2, x3, x4], axis=1), axis=1).astype(float)
        ymax = np.max(np.stack([y1, y2, y3, y4], axis=1), axis=1).astype(float)
	
        assert len(xmin.shape) == 1, "Wrong shape x1: {}".format(x1.shape)
        areas = (xmax - xmin + 1) * (ymax - ymin + 1)
        areas = areas.astype(float)

        order = areas.argsort()[::-1]
        keep = []

        while order.size > 0:
            i = order[0]
            keep.append(i)

            order = order[1:]

            xx1 = np.maximum(xmin[i], xmin[order])
            yy1 = np.maximum(ymin[i], ymin[order])
            xx2 = np.minimum(xmax[i], xmax[order])
            yy2 = np.minimum(ymax[i], ymax[order])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / areas[order]

            inds = np.where(ovr <= thresh)[0]
            order = order[inds]

        nms_boxes = boxes[keep, :, :].copy()
        nms_scores = scores[keep].copy()
		
        return nms_boxes, nms_scores

    def inference(self, image_path, visualize=False):
        self.init_torch_tensor()

        # Init models
        db_model, dbplus_model = self.init_model()

        # Resume models
        self.resume(db_model, self.db_model_path)
        self.resume(dbplus_model, self.dbplus_model_path)

        all_matrics = {}

        # Switch to eval mode
        db_model.eval()
        dbplus_model.eval()

        # Infer
        batch = dict()
        batch['filename'] = [image_path]
        img, original_shape = self.load_image(image_path)

        batch['shape'] = [original_shape]
        with torch.no_grad():
            batch['image'] = img
            db_pred = db_model.forward(batch, training=False)
            db_output = self.db_structure.representer.represent(batch, db_pred, is_output_polygon=self.args['polygon'])

            dbplus_pred = dbplus_model.forward(batch, training=False)
            dbplus_output = self.dbplus_structure.representer.represent(batch, dbplus_pred, is_output_polygon=self.args['polygon'])

            # Format output
            if not os.path.isdir(self.args['result_dir']):
                os.mkdir(self.args['result_dir'])
            output = self.format_output(batch, db_output, dbplus_output)

            # Visualize using db structure
            if visualize and self.dbplus_structure.visualizer:
                vis_image = self.dbplus_structure.visualizer.demo_visualize(image_path, output)
                cv2.imwrite(os.path.join(self.args['result_dir'], image_path.split('/')[-1].split('.')[0] + '.jpg'),
                            vis_image)

            predict_path = os.path.join(self.args['result_dir'],
                                        'res_' + image_path.split('/')[-1].split('.')[0] + '.jpg').replace('jpg', 'txt')
            new_lines = ''
            
            with open(predict_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    coor = [int(i) for i in line.split(',')[:8]]
                    for i in range(4):
#                        coor[2 * i + 1] = min(vis_image.shape[0] - 2 * self.args['margin'], max(0, coor[2 * i + 1] - self.args['margin']))
						# Constant padding
                        coor[2 * i + 1] = min(vis_image.shape[0] - 2 * self.args['margin_h'], max(0, coor[2 * i + 1] - self.args['margin_h']))
                        coor[2 * i] = min(vis_image.shape[1] - 2 * self.args['margin_w'], max(0, coor[2 * i] - self.args['margin_w']))

						# Scale padding
                        pad_size = [int(float(vis_image.shape[0]) / (1 + 2 * self.args['margin_scale_h']) * self.args['margin_scale_h']),int(float(vis_image.shape[1]) / (1 + 2 * self.args['margin_scale_w']) * self.args['margin_scale_w'])] 

                        coor[2 * i + 1] = min(vis_image.shape[0] - 2 * pad_size[0], max(0, coor[2 * i + 1] - pad_size[0]))
                        coor[2 * i] = min(vis_image.shape[1] - 2 *pad_size[1], max(0, coor[2 * i] - pad_size[1]))

                    coor = [str(i) for i in coor]
                    coor = ','.join(coor)
                    coor += ',' + ','.join(line.split(',')[8:])
                    new_lines += coor
            with open(predict_path, 'w') as f:
                f.write(new_lines)


if __name__ == '__main__':
    main()

