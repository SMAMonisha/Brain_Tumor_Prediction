from tqdm.auto import tqdm

import os
import requests
import zipfile
import cv2
import math
import matplotlib.pyplot as plt
import glob
import numpy as np
import random
import seaborn as sns
import pandas as pd 

from PIL import Image
import yaml

import random
import cv2
import os
import matplotlib.pyplot as plt

image_dir = 'E:/BioM/archive/BrainTumor/BrainTumorYolov11/train/images'
label_files = 'E:/BioM/archive/BrainTumor/BrainTumorYolov11/train/labels'

image_files = os.listdir(image_dir)
random_images = random.sample(image_files, 40)

fig, axs = plt.subplots(8, 5, figsize=(22, 22))  
axs = axs.flatten() 

for i, image_file in enumerate(random_images):
    image_path = os.path.join(image_dir, image_file)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  

    label_file = os.path.splitext(image_file)[0] + ".txt"
    label_path = os.path.join(label_files, label_file)
    
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            labels = f.read().strip().split("\n")

        for label in labels:
            if len(label.split()) != 5: 
                continue
            class_id, x_center, y_center, width, height = map(float, label.split())
            x_min = int((x_center - width / 2) * image.shape[1])
            y_min = int((y_center - height / 2) * image.shape[0])
            x_max = int((x_center + width / 2) * image.shape[1])
            y_max = int((y_center + height / 2) * image.shape[0])

            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)

            text_scale = max(0.5, min(image.shape[1], image.shape[0]) / 700)
            text_thickness = max(1, int(text_scale * 2))
            classe = class_names[int(class_id)]  
            cv2.putText(
                image,
                classe,
                (x_min, max(y_min - 10, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                text_scale,
                (0, 0, 255),
                text_thickness,
                cv2.LINE_AA,
            )
    
    axs[i].imshow(image)
    axs[i].axis('off')

from ultralytics import YOLO
model=YOLO('yolo11s.pt')

model.train(
   data='E:/BioM/archive/BrainTumor/BrainTumorYolov11/data.yaml',
   imgsz=64,
   epochs=10,
   batch=256)

