
# https://medium.com/@huzeyfebicakci/custom-dataset-training-with-yolov10-a-deep-dive-into-the-latest-evolution-in-real-time-object-ab8c62c6af85
# modified by Alfonso Blanco García
import os
import cv2
import supervision as sv
import ultralytics
# instalado con !pip install -q git+https://github.com/THU-MIG/yolov10.git
# https://blog.roboflow.com/yolov10-how-to-train/
# en mi caso:
#(alfonso1) C:\Users\Alfonso Blanco\.conda\envs\alfonso1\Scripts>python pip-script.py install git+https://github.com/THU-MIG/yolov10.git
from ultralytics import YOLOv10
import torch

class ObjectDetection:
    def __init__(self):
        #home_dir = os.path.expanduser('~')
        #self.dir = os.path.join(home_dir, "Desktop", "yolo_v10") 
        self.dir="C:/Fracture.v1i_Reduced_Yolov10"
    def train(self, runName):
        # downloaded from https://github.com/THU-MIG/yolov10/releases
        model = YOLOv10("yolov10n.pt")
        
        #yaml_path = os.path.join(self.dir, 'yaml_file.yaml')
        yaml_path = "FractureYolov10OJumbo1.yaml"
        results = model.train(
            data= yaml_path,         # Path to your dataset config file
            batch = 16,               # Training batch size
            imgsz= 640,                   # Input image size
            #epochs= 2000,                  # Number of training epochs
            epochs= 200,                  # Number of training epochs
            optimizer= 'SGD',             # Optimizer, can be 'Adam', 'SGD', etc.
            lr0= 0.01,                    # Initial learning rate
            
            lrf= 0.1,                     # Final learning rate factor
            
            weight_decay= 0.0005,         # Weight decay for regularization
            momentum= 0.937,              # Momentum (SGD-specific)
            verbose= True,                # Verbose output
            #device= '0',                  # GPU device index or 'cpu'
            device= 'cpu',                  # GPU device index or 'cpu'
            workers= 8,                   # Number of workers for data loading
            project= 'runs/train',        # Output directory for results
            name= 'exp',                  # Experiment name
            exist_ok= False,              # Overwrite existing project/name directory
            rect= False,                  # Use rectangular training (speed optimization)
            resume= False,                # Resume training from the last checkpoint
            #multi_scale= False,           # Use multi-scale training
            multi_scale= True,
            #single_cls= False,             # Treat data as single-class
            single_cls= False, 
            #freeze = 20, #default değer : none
            #resume=True, #Başka bilgisatarda eğitim devamı yapılamıyor.
            #name=runName,
        )

    @staticmethod
    def train_custom_dataset(runName):
        od = ObjectDetection()
        od.train(runName)


# Example usage:
ObjectDetection.train_custom_dataset('trained_model')
