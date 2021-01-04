#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 08:55:04 2021

@author: sprasad
"""

import cv2
import numpy as np 
import argparse
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# import time
from moviepy.editor import VideoFileClip
import glob



parser = argparse.ArgumentParser()
parser.add_argument('--webcam', help="True/False", default=False)
parser.add_argument('--play_video', help="Tue/False", default=False)
parser.add_argument('--image', help="Tue/False", default=False)
parser.add_argument('--video_path', help="Path of video file", default="videos/car_on_road.mp4")
parser.add_argument('--image_path', help="Path of image to detect objects", default="Images/bicycle.jpg")
parser.add_argument('--verbose', help="To print statements", default=True)
args = parser.parse_args()

#Load yolo
def load_yolo():
	net = cv2.dnn.readNet("./yolo_data/yolov3.weights", "./yolo_data/yolov3.cfg")
	classes = []
	with open("./yolo_data/coco.names", "r") as f:
		classes = [line.strip() for line in f.readlines()]

	layers_names = net.getLayerNames()
	output_layers = [layers_names[i[0]-1] for i in net.getUnconnectedOutLayers()]
	colors = np.random.uniform(0, 255, size=(len(classes), 3))
	return net, classes, colors, output_layers

def load_image(img_path):
	# image loading
	img = cv2.imread(img_path)
	img = cv2.resize(img, None, fx=0.4, fy=0.4)
	height, width, channels = img.shape
	return img, height, width, channels


def detect_objects(img, net, outputLayers):			
	blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(320, 320), mean=(0, 0, 0), swapRB=True, crop=False)
	net.setInput(blob)
	outputs = net.forward(outputLayers)
	return blob, outputs

def get_box_dimensions(outputs, height, width):
    boxes = []
    confs = []
    class_ids = []
    for output in outputs:
        for detect in output:
            scores = detect[5:]
            class_id = np.argmax(scores)
            conf = scores[class_id]
            if conf > 0.3:
                center_x = int(detect[0] * width)
                center_y = int(detect[1] * height)
                w = int(detect[2] * width)
                h = int(detect[3] * height)
                x = int(center_x - w/2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confs.append(float(conf))
                class_ids.append(class_id)
    return boxes, confs, class_ids
			
def draw_labels(boxes, confs, colors, class_ids, classes, img): 
    indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.5, 0.3)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            # color = colors[0]
            color = np.array([255.0,0.0,0.0])
            cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
            cv2.putText(img, label, (x, y - 5), font, 1, color, 1)
    cv2.imshow("Image", img)

def image_detect(image): 
    # model, classes, colors, output_layers = load_yolo()
    # image, height, width, channels = load_image(img)
    height, width, channels = image.shape
    blob, outputs = detect_objects(image, model, output_layers)
    boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
    draw_labels(boxes, confs, colors, class_ids, classes, image)
# 	while True:
# 		key = cv2.waitKey(1)
# 		if key == 27:
# 			break
    return image
# %%
frame_nr = 149
test_image = mpimg.imread('./my_test_images/frame' + str(frame_nr) + '.jpg')
model, classes, colors, output_layers = load_yolo()
test_image_output = image_detect(test_image)
plt.imshow(test_image_output)

# %%
test_img_paths = glob.glob('./my_test_images/frame*.jpg')

test_imgs = []
test_imgs_output = []

for idx, path in enumerate(test_img_paths):
    test_imgs.append(mpimg.imread(path))
    test_imgs_output.append(image_detect(test_imgs[idx]))
    
# %%
f1, ax1 = plt.subplots(3, 2, figsize=(15, 18))
f1.tight_layout()
for ax2, idx in zip(ax1.flat, range(6)):
    ax2.imshow(test_imgs_output[idx])
    ax2.set_title(test_img_paths[idx], fontsize=14)
    ax2.axis('off')
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
f1.savefig('./my_output_images/frame_pipeline_yolo_output.jpg')
# %%
test_out_file = 'my_test_video_3_output_yolo3.mp4'
clip_test = VideoFileClip('my_test_video_3.mp4')
clip_test_out = clip_test.fl_image(image_detect)
clip_test_out.write_videofile(test_out_file, audio=False)
