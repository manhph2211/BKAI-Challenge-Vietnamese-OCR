BKAI-NAVER Challenge 2022 - Vietnamese Scene Text Detection and Recognition 
====

# Giới thiệu :smile:

- Cuộc thi BKAI-NAVER Challenge 2022 được tổ chức tại Đại Học Bách Khoa Hà Nội với 3 tracks. Trong đó, track 3 sẽ được thực hiện bởi nhóm trong repo này. Nhóm đã đạt được kết quả khá tốt 😸, kết quả của nhóm còn có thể tốt hơn nhiều nếu đầy đủ sức mạnh tài nguyên :smile: . Đây thực sự là một trải nghiệm tuyệt vời <3
<!-- ![image](https://user-images.githubusercontent.com/61444616/169662978-1faf6fe5-5594-4198-870e-6b415ddfa52b.png) -->

- Team với các thành viên Phú, Đức Anh và Mạnh:
<p align="center">
  <img src="https://user-images.githubusercontent.com/61444616/169662957-8221f7fd-799b-44d5-9f4b-788cf2637a93.png" width="800">
</p>

- Mô tả pipeline: 
  - Dựa theo yêu cầu của bài toán này thì có hai bước không thể thiếu ở đây là `Text Detection` và `Text Recognition`. Thực tế, một số cách tiếp cận hiện đại đã cố gắng để gộp hai phần này lại với nhau, tuy nhiên điều này không thực sự mang lại kết quả có thể so sánh được với việc tiếp cận hai task trên một cách độc lập hoàn toàn. 
  - Ở đây, sau khi nhận dữ liệu được cấp từ ban tổ chức, chúng em đã đầu tiên chuẩn bị lại dữ liệu để sẵn sàng đưa vào huấn luyện cho task Text Dection, mà ở đây, chúng em đã dùng DB. Cụ thể, dữ liệu được cấp có 2500 ảnh có nhãn và 235 ảnh trong tập public test. Chúng em chia 2500 ảnh có nhãn ra thành hai tập train và val, trong đó tập train sử dụng 2200 ảnh. Trong quá training, DB đã được pretrained trên tập dataset Totaltext bao gồm 1555 ảnh với đa dạng các phông nền cũng như độ cong văn bản, sẽ tiếp tục training với 2200 ảnh của ban tổ chức. Ở đây, chúng em nhận thấy mô hình DB được sử dụng có phần nhạy cảm trong quá trình inference, khi mà gặp phải những ảnh với chiều cao nhỏ, chỉ có một dòng text, và đặc biệt là hai tham số thresolds đặc trưng của mô hình. Vì vậy trong quá trình inference, chúng em đề xuất sử dụng thêm kỹ thuật padding chiều cao và chiều rộng một khoảng nhỏ cho ảnh ngay trước khi cho mô hình dự đoán. Chúng em tiếp tục tận dụng sức mạnh của model DBNet++ để ensemble với DB hiện tại ở range thresold khác nhau, sau đó sử dụng kỹ thuật NMS kết hợp một kĩ thuật dựa trên diện tích box để loại bỏ các box chồng lấn nhiều lên nhau loại sau đó merge box đáng nhẽ phải thuộc về nhau. 
  - Sau khi đã có được bounding box của các đoạn văn bản trong bức ảnh, chúng em tiến hành cắt phần text ra khỏi ảnh, đồng thời ở đây, với mong muốn có một tập dữ liệu tốt hơn cho task sau, chúng em tiến hành căn chỉnh lại phần ảnh văn bản này sao cho những văn bản bị nghiêng sẽ được hiệu chỉnh lại cho nằm ngang. Tiếp theo, để phục vụ cho việc sinh thêm dữ liệu training cho task sau, chúng em đã tận dụng những background có sẵn ở trong tập training này, bằng cách loại bỏ các phần pixel là văn bản trong ảnh gốc dựa trên một kỹ thuật đơn giản về việc hiệu chỉnh pixel. 
  - Ở task text recognition, chúng em đề xuất sử dụng mô hình VietOCR với ý tưởng dựa trên Transformer OCR, đã được chứng minh là có hiệu quả tốt với tiếng Việt. Ở đây, ảnh văn bản được cắt ra từ tập dữ liệu ban tổ chức cung cấp, bao gồm khoảng 25000 ảnh, kết hợp với bộ dữ liệu mà bọn em sinh ra, bao gồm 20000 ảnh được sinh với các đa dạng các loại font chữ cũng như lấy backgrounds từ bước trước đó. Kết quả dự đoán từ task này sẽ được tổng hợp với task Text Detection và từ đó chúng em tạo folder prediction như theo format mà ban tổ chức đã đưa ra. Dưới đây là pipeline chính của chúng em cho challenge này. 
 
<p align="center">
  <img src="https://user-images.githubusercontent.com/61444616/169663007-fdc470a0-e858-440d-9478-03c9dd676de9.png" width="800">
</p>

# Chuẩn bị dữ liệu 
- Tập dữ liệu public và private có tải về từ <a href="https://drive.google.com/drive/folders/1WNOk3EMSgawdbrHeLiO7N_WcapgmbdmV">dataset 1</a>, <a href="https://drive.google.com/file/d/1UUQhNvzgpZy7zXBFQp0Qox-BBjunZ0ml/view?usp=sharing">dataset 2</a>, <a href="https://drive.google.com/drive/folders/14jV4W7Cdz5Q-znwhY0GenLrWzKbgS4ah?usp=sharing">public test</a>.

- Unzip file `vietnamese_original.zip`, `train_imgs.zip`, `train_gt.zip`, sau đó lưu vào trong folder `src/data`

- Đổi tên file data: (Nhóm đã thực hiện đổi tên về dạng index.jpg, index.txt (index từ 0 đén 2499) và lưu vào trong `data.zip`)

- Run các lệnh:

```
$ python data_prepair.py
$ cd TransOCR-Pytorch/src/dataset
$ python utils_data.py
```

# Setup môi trường 

- Chúng ta có thể sử dụng conda ở đây cho việc cài đặt môi trường và các required packages

```
$ conda create -n BKAI python=3.9
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
- Các bạn có thể tham khảo [checkpoint](https://drive.google.com/file/d/190ruQxYy7xvpjutmgJ6HDJ26xdG_dv5s/view?usp=sharing) do nhóm mình training trong cuộc thi.

# Kết luận

- Nhóm đã hoàn thiện pipeline và demo với kết quả khá tốt!

# Hướng phát triển

- Gen thêm scene text data, training thêm cả 2 task với dữ liệu lớn
- Thử nghiệm mô hình Craft để kết hợp trong task 1
- Triển khai ensemble task 2 với ABCNet và SRN, đong thời xem xét bước hậu xử lý text (auto correct some cases,...)
- Tối ưu hyper parameters( đặc biệt là các thresolds, ...)
- Viết Flask API để deploy hệ thống

# Tài liệu tham khảo

- [DBNet](https://github.com/MhLiao/DB)
- [VietOCR](https://github.com/pbcquoc/vietocr)
- [Scene Text Gen](https://github.com/Belval/TextRecognitionDataGenerator)
- [Data augmentation](https://github.com/aleju/imgaug)
