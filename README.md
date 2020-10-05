# DVC
Caffe implementation of our work entitled as "Deep Video Code for Efficient Face Video Retrieval". For research use only, commercial use is not allowed.
Our previous conference released codes, models and datasets are placed [here](https://pan.baidu.com/s/1o7HdzGu) (BaiduCloud extracted codes: ugqs). This repository 
is an improved implementation in our Pattern Recognition work (under review), which support videos with varied lengths and multimodal data (image and video).   

An illustration of the retrieval task as follows:
![image](https://github.com/greatmanqss/DVC/blob/master/FaceVideoRetrieval.png)

And the framework of our proposed method DVC:
![image](https://github.com/greatmanqss/DVC/blob/master/DVC_framework.png)

# Citation
If you use this code for your research, please consider citing our paper: Shishi Qiao, Ruiping Wang, Shiguang Shan, Xilin Chen. Deep Video Code for Efficient Face Video Retrieval. ACCV 2016. (PR under review) 

# Prerequisites
Linux 14.04 (We simply tried it on 16.06 or high version but failed due to MKL issue)

NVIDIA GPU + CUDA-7.5 or CUDA-8.0 and corresponding CuDNN

Caffe

BLAS lib: Intel MKL V2017.1.132

# Modifications on Caffe
- Add convert_imageset_set in the tools which converts video clips into lmdb format
- Add extract_features_binary in the tools which extracts the outputs of one layer of a trained model into binary file
- Modified db, db_leveldb, db_lmdb, data_reader, data_layer which deal with the image and video data in lmdb format during training and testing 
- Modified math_functions in the utils which now supports the svd and more matrix operations with the help of MKL BLAS
- Add sub_mean_layer, covlogm_layer, temporal_pooling_layer which handle the video modeling procedure for face videos
- Add bound_triplet_loss and other metric learning loss which are used for hashing supervision
- Modified caffe.proto to support corresponding modifications listed above

# Compiling
The compiling process is the same as caffe. You can refer to Caffe installation instructions [here](http://caffe.berkeleyvision.org/installation.html).

# Datasets
We use BBT, PB, YTC and a subset containing 200 subjects of UMDFaces dataset in our experiments. We have preprocessed these three datasets and provided both the raw images and the converted lmdb files for direct training and testing. You can download them [here](https://pan.baidu.com/s/1mWgt7ufaoJ88iWk1vkuSXQ) and [here](https://pan.baidu.com/s/1o7HdzGu) using the extracted codes：m0d9 and ugqs (BaiduCloud drive). And in the future, we will provide a download link on google drive.

After downloading, you can directly use the lmdb files for training and testing DVC. Also you can convert the raw images together with split txt files to the LMDB format as we have provided for you.
For video modality, you can use the following command for YTC dataset as an example to convert the video clips:
```
./build/tools/convert_imageset_set --resize_height=64 --resize_width=64 path/to/orig_imgs_folder/  /path/to/train_shuffle.txt or test_shuffle.txt   path/to/train_test_fold    path/to/saved lmdb file
```
For image modality, you can use the following command for YTC dataset as an example to convert the still images:
```
./build/tools/convert_imageset --resize_height=64 --resize_width=64 path/to/orig_imgs_folder/    path/to/train_still.txt or test_still.txt    path/to/saved lmdb file 
```

# Training
We place the solver and net prototxt files in the examples/DVC_journal/YTC folder. First, you need to download the pre-trained classification model [here](https://pan.baidu.com/s/1mWgt7ufaoJ88iWk1vkuSXQ) using the extracted codes：m0d9 (BaiduCloud drive) for initilizing DVC and move it to ./models/. Then, you need to modify the corresponding paths in the solver and net prototxt files. Finaly, you can train DVC for each dataset using the followling command (here we use YTC as an example):
```
./build/tools/caffe train --solver ./examples/DVC_journal/YTC/casia_solver.prototxt --weights ./models/YTC/ytc_classification_iter_5000.caffemodel
```


# Evaluation
You can evaluate the mean Average Precision(mAP) result on each dataset. First, you need to extract the binary codes and labels of videos and images using the following command (YTC as an example):
```
./build/tools/extract_features_binary   path/to/trained DVC models    ./examples/DVC_journal/YTC/casia_train_val.txt    ip1 (hash layer output of videos)    path/to/saved file     batch_num    GPU id

./build/tools/extract_features_binary   path/to/trained DVC models    ./examples/DVC_journal/YTC/casia_train_val.txt    merge_label (labels of videos)    path/to/saved file     batch_num    GPU id  
```

And then you use the extracted binary files to evaluate mAP results
# Contact
If you have any problem about our code, feel free to contact shishi.qiao@vipl.ict.ac.cn, qiaoshishi14@mails.ucas.ac.cn or describe your problem in Issues.
