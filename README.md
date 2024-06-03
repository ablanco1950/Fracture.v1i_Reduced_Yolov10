# Fracture.v1i_Reduced_Yolov10
From a selection of data from the Roboflow file https://universe.roboflow.com/landy-aw2jb/fracture-ov5p1/dataset/1, which represents a reduced but homogeneous version of that file, a model is obtained based on yolov10 with that custom dataset to indicate fractures in x-rays.

By using only 147 images, training is allowed using a personal computer without GPU

===
Facility:

Download all project datasets to a folder on disk.

Install yolov10 following the instructions given at:

https://blog.roboflow.com/yolov10-how-to-train/

which are reduced to

 !pip install -q git+https://github.com/THU-MIG/yolov10.git

And download from https://github.com/THU-MIG/yolov10/releases the yolov10n.pt model. In case this operation causes problems, this file is attached with the rest of the project files.

To download the dataset you have to register as a roboflow user, but to simplify it, the compressed files for train, valid and test are attached: trainFractureOJumbo1.zip, validFractureOJumbo1.zip and testFractureOJumbo1.zip obtained by selecting the images that start with names 0 -_Jumbo-1 of the original file obtaining a reduced number of images that allow training with yolov10 on a personal computer without GPU

Some zip decompressors duplicate the name of the folder to be decompressed; a folder that contains another folder with the same name should only contain one. In these cases it will be enough to cut the innermost folder and copy it to the project folder.

===
Proof:

It is executed:

EvaluateTESTFractureYolov10.py

The x-rays are presented on the screen with a blue box indicating the prediction and a green box indicating the label. The console indicates the images in which no fracture has been detected.
Of the 9 images, 6 appear correctly detected and marked and 3 could not be detected.
The results are much lower than those obtained by testing the 9 images with https://universe.roboflow.com/landy-aw2jb/fracture-ov5p1/model/1 where only 1 is not detected

===
 Training

The project comes with an optimized model: last128epoch060003.pt

To obtain this model, the following has been executed:

EvaluateTESTFractureYolov10.py

which uses the .yaml file:

FractureYolov10OJumbo1.yaml

File with the Log is attached:

LOG_200epochMultiScale.txt

The training results with the best.pt and last.pt models are obtained at the address:
runs\train\exp\weights

With the following warnings:

Both in EvaluateTESTFractureYolov10.py (line 19) and in FractureYolov10OJumbo1.yaml the absolute addresses of the project appear assuming that it has been installed on disk C:, if it has another location these absolute addresses will have to be changed.

The best result has been found in the last.pt of 128 epoch, although when reaching the end, in 200 epoch, it seems that the values ​​of mAP50 and mAP50-95 are better. Which would indicate that the training has overfitting

===
References and citations:



https://blog.roboflow.com/yolov10-how-to-train/
James Gallagher, Piotr Skalski. (May 24, 2024). How to Fine-Tune a YOLOv10 Model on a Custom Dataset. Roboflow Blog: https://blog.roboflow.com/yolov10-how-to-train/