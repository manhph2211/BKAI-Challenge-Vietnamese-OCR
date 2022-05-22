1. unzip file vietnamese_original.zip, train_imgs, train_gt.zip, sau đó lưu vào trong folder src/data
2. Đồi tên file data: (nhom da thưc hiện đổi tên về dạng index.jpg, index.txt (index từ 0 đén 2499) và lưu vào trong data.zip)
$ python data_prepair.py
$ cd TransOCR-Pytorch/src/dataset
$ python utils_data.py
3. Khởi tạo môi trường
$ conda activate DB
3. Cài requirements.txt
$ pip install -r requirements.txt
4. Train DB
$ cd DB
- Tạo folder datasets: $ ln -s ../data datasets
- Chia train/test: $ python data_prepair.py
- Tair pretrained: totaltext_resnet50_defor và lưu vào trong folder DB/pretrained
link: https://drive.google.com/drive/folders/1T9n0HTP3X3Y_nJ0D1ekMhCQRHntORLJG
- Training
Voi DBNet
$ CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py experiments/seg_detector/totaltext_resnet50_deform_thre.yaml --resume pretrained path_to_pretrained
Voi DBNet++
$ CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py experiments/ASF/totaltext_resnet50_deform_thre.yaml --resume pretrained path_to_pretrained
Nhóm đã train và đạt kết quả tốt, lưu checkpoint ở trong DB/workspace/SegDetectorModel-seg_detector/L1BalanceCeLoss/model/model_epoch_66_minibatch_4500
5. Text generator
Chuyển TextRecognitionDataGenertor từ data.zip sang src/
$ mv data/TextRecognitionDataGenertor BKAI
$ cd BKAI/TextRecoginitionDataGenerator
$ python create_background.py
Nhóm đã tập file custom.txt tiếng Việt dựa tren file vn_dictionary.txt mà ban tổ chức cung cấp bằng file clean_dict.py
Sau đó, nhóm đã lứa chọn fonts phù hợp cho tiếng Việt, tat cả trong trgd/fonts/custom
$ python run.py -c 20000 -k 15 -rk -rbl -tc '#000000,#888888' --fit -i path_to_dictionary --output_dir path_to_custom_gen_data  -na 2 -fd path_to_vietnames_font_dir -id path_to_backround_dir
path_to_dictionary : là path tới file custom.txt
path_to_font : trgd/fonts/custom
path_to_backgourd : trgd/images
path_to_custom : trong vietnamese/custom_gen_data
$ cd data/custom_data/
$ python custom.py
5. Tao dataset cho VietOCR
$ cd TransOCR-Pytorch/src/utils
$ python create_dataset.py
6. Train VietOCR
$ cd TranssOCR-Pytorch/tools
$ CUDA_VISIBLE_DEVICES=0,1 train.py
7. demo
7.1. DB
$ cd DB
- Khong su dung ensemble:
$ CUDA_VISIBLE_DEVICES=0 python demo.py experiments/seg_detector/totaltext_resnet50_deform_thre.yaml --resume workspace/SegDetectorModel-seg_detector/deformable_resnet50/L1BalanceCELoss/model/model_epoch_66_minibatch_4500 --image_path ../data/public_test_img --visualize --box_thresh 0.35 --margin_h 40 --margin_w 30 --thresh 0.28
- Su dung ensemble:
$ CUDA_VISIBLE_DEVICES=0 python esemble_demo.py --exp_db experiments/seg_detector/totaltext_resnet50_deform_thre.yaml --exp_dbplus experiments/ASF/totaltext_resnet50_deform_thre_asf.yaml --checkpoint_dbplus workspace/SegDetectorModel-seg_detector/deformable_resnet50/L1BalanceCELoss/model/final --checkpoint_db workspace/SegDetectorModel-seg_detector/deformable_resnet50/L1BalanceCELoss/model/DBNet/model_epoch_66_minibatch_4500 --image_path /home/edabk/phumanhducanh/BKAI/private_test --visualize --box_thresh 0.35 --margin_scale_h 0.25 --margin_scale_w 0.3 --thresh 0.25
7.2. TransOCR
$ cd TransOCR-Pytorch/tools
$ CUDA_VISIBLE_DEVICES=0 python submit.py
