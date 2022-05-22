BKAI-NAVER Challenge 2022 - Vietnamese Scene Text Detection and Recognition :smiley: 
====

# Giới thiệu :smile:

- Cuộc thi BKAI-NAVER Challenge 2022 được tổ chức tại Đại Học Bách Khoa Hà Nội với 3 tracks. Trong đó, track 3 sẽ được thực hiện bởi nhóm trong repo này. Nhóm đã đạt được kết quả LOL và xếp hạng thứ LOL. Đây thực sự là một trải nghiệm tuyệt vời <3
<!-- ![image](https://user-images.githubusercontent.com/61444616/169662978-1faf6fe5-5594-4198-870e-6b415ddfa52b.png) -->

- Team với các thành viên Phú, Đức Anh và Mạnh:
<img src="https://user-images.githubusercontent.com/61444616/169662957-8221f7fd-799b-44d5-9f4b-788cf2637a93.png" width="800">

- Mô tả pipeline
<img src="https://user-images.githubusercontent.com/61444616/169663007-fdc470a0-e858-440d-9478-03c9dd676de9.png" width="800">

# Chuẩn bị dữ liệu 

- Unzip file `vietnamese_original.zip`, `train_imgs.zip`, `train_gt.zip`, sau đó lưu vào trong folder `src/data`

- Đồi tên file data: (nhom da thưc hiện đổi tên về dạng index.jpg, index.txt (index từ 0 đén 2499) và lưu vào trong `data.zip`)

- Run các lệnh:

```
$ python data_prepair.py
$ cd TransOCR-Pytorch/src/dataset
$ python utils_data.py
```

# Setup môi trường 

- Chúng ta có thể sử dụng conda ở đây cho việc cài đặt môi trường và các required packages

```
$ conda create -n BKAI python=3.9.12
$ conda activate BKAI
$ pip install -r requirements.txt
```

# Quá trình tiến hành 

## Text Detection
 
- Đầu tiên, chúng ta cần:

```
$ cd DB
$ ln -s ../data datasets # Tạo folder datasets: 
$ python data_prepair.py # Chia train/test: 
```

- Tiếp theo, chúng ta sẽ Tair pretrained: [totaltext_resnet50_defor](https://drive.google.com/drive/folders/1T9n0HTP3X3Y_nJ0D1ekMhCQRHntORLJG) và lưu vào trong folder `DB/pretrained`

- Training DB, ở đây nhóm đã train và đạt kết quả tốt tại `epoch 66`, lưu ở trong `DB/workspace/SegDetectorModel-seg_detector/L1BalanceCeLoss/model/model_epoch_66_minibatch_4500`

```
$ CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py experiments/seg_detector/totaltext_resnet50_deform_thre.yaml --resume pretrained path_to_pretrained
```

## Text generator

- Chuyển `TextRecognitionDataGenertor` từ `data.zip` sang `src/`

```
$ mv data/TextRecognitionDataGenertor BKAI
$ cd BKAI/TextRecoginitionDataGenerator
$ python create_background.py
```

- Nhóm đã tạo file `custom.txt` tiếng Việt dựa trên file `vn_dictionary.txt` mà ban tổ chức cung cấp bằng file `clean_dict.py`

- Sau đó, nhóm đã lứa chọn `fonts` phù hợp cho tiếng Việt, tất cả trong `trgd/fonts/custom`

- Tiến hành run các câu lệnh sau để sinh `20000` ảnh cho task này:

```
$ python run.py -c 20000 -k 15 -rk -rbl -tc '#000000,#888888' --fit -i path_to_custom.txt --output_dir vietnamese/custom_gen_data  -na 2 -fd trgd/fonts/custom -id trgd/images
$ cd data/custom_data/
$ python custom.py
```

- Tạo dataset cho VietOCR

```
$ cd TransOCR-Pytorch/src/utils
$ python create_dataset.py
```

- Train VietOCR

```
$ cd TranssOCR-Pytorch/tools
$ CUDA_VISIBLE_DEVICES=0,1 train.py
```

## Demo 

- Cuối cùng để demo, chúng ta tiến hành gõ các lệnh lần lượt:

```
$ cd DB
$ CUDA_VISIBLE_DEVICES=0 python esemble_demo.py --exp_db experiments/seg_detector/totaltext_resnet50_deform_thre.yaml --exp_dbplus experiments/ASF/totaltext_resnet50_deform_thre_asf.yaml --checkpoint_dbplus workspace/SegDetectorModel-seg_detector/deformable_resnet50/L1BalanceCELoss/model/final --checkpoint_db workspace/SegDetectorModel-seg_detector/deformable_resnet50/L1BalanceCELoss/model/DBNet/model_epoch_66_minibatch_4500 --image_path ../private_test --visualize --box_thresh 0.35 --margin_scale_h 0.25 --margin_scale_w 0.3 --thresh 0.25
$ cd TransOCR-Pytorch/tools
$ CUDA_VISIBLE_DEVICES=0 python submit.py
```

# Kết luận

# Hướng phát triển

# Tài liệu tham khảo
