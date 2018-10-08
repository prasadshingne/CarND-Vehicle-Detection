**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/veh_nonveh_example.jpg
[image2]: ./output_images/veh_HOG_example.jpg
[image3]: ./output_images/nonveh_HOG_example.jpg
[image4]: ./output_images/test4_all_boxes.jpg
[image5]: ./output_images/test4_boxes_raw.jpg
[image6]: ./output_images/test4_heat_box.jpg
[image7]: ./output_images/test4_thresholded.jpg
[image8]: ./output_images/test4_boxes_processed.jpg
[image9]: ./output_images/frame_pipeline_output.jpg

[video1]: ./project_out_frame3.mp4
[video2]: ./project_output_smooth_3.mp4


## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

I read in the training data set in cell 2. I randomly select and display a vehicle and non-vehicle image in cell 3. There are 8792 vehicle images and 8968 non-vehicle images in the training data set.   

Here is an example of one of each of the `vehicle` and `non-vehicle` classes:
![alt text][image1]

Cell 4 shows functions taken from lesson materials that return spatial, color and hog feature extraction. Cell 5 shows a modified version of the feature extraction function from the lesson wich extract spatial, color and hog features based on input flags. Following figure shows the HOG features for vehicle and non-vehicle images previously shown (cell 8).

![alt text][image2]
![alt text][image3]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried a number of differecnt combinations of the orientations, pixels per cell, cells per block and HOG channels for feature extraction. I selected these along with color and spatial feature parameters that gave the best fit. Paramertes are presented below. 

I picked the following parameters (cells 6 and 7) that produced a high test accuracy while keeping the feature extraction and fitting time reasonably low. 

| Parameter     | Value         | 
|:-------------:|:-------------:| 
| Colorspace     | YUV      | 
| HOG orient    | 14     |
| HOG pixels per cell    |16   |
| Hog cells per block     | 4       |
| Hog channels     | All       |
| Histogram bins    | 32      |
| Histogram range   | (0, 256)     |
| Spatial bin size  | (16,16)      |
| Classifier  | LinearSVC      |
| Scalar  | StandardScaler     |

These parameters led to a feature length of 1536 with a feature extraction time of ~90 seconds, training time of 4.5 seconds and test accuracy of 98.03%.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

In cell 7 I trained a linear SVM using HOG, color and spatial features to get a test accuracy of 98.03%.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I implemented the more efficient sliding window search approach based on the lesson material (Cell 9). Here we extract HOG features are extracted just once for each of the small set of predetermined window sizes based on the scale argument and then can be subsampled for overlaying windows. The function returns a list of rectangle objects corresponding to the windows that generated 'car' prediction. I run the same function four times (cell 11) for different configurations of (y_start, y_stop) and scale. After some trial and error the following configurations are used to aggregate all rectangles that made a 'car' prediction.

|Scale     | Y_start    |  Y_stop |
|:--------:|:----------:|:-------:|
|1         |380         | 650     |
|1.5       |380         | 600     |
|2.0       |400         | 650     |
|2.5       |400         | 660     |
|3.0       |350         | 680     |

Following picture shows the four configuration schemes of search windows on test image 4.

![alt text][image4]

Following image shows all the rectangles returned based on the above configuration. There are multiple positive predictions for each of the cars close by whereas few predictions for the oncoming cars to the left that we are not interested in.

![alt text][image5]

Following figure shows a heat map corresponding to the previous image where tbe function add_heat (cell 14) increments the pixel value on an all black image of the size of the original image at the location of each rectangle detected. Regions covered by more overlapping rectangles result in higher level of 'heat'.

![alt text][image6]

A threshold is applied to the heatmap (cell 17), with a value of 2.0 in this case, setting all pixels with value less than this to zero. Following figure shows the result corresponding to the label identified after thresholding the previous image.

![alt text][image7]


#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Folloing figure shows the result of passing the test images though my pipeline. Note that my pipline does not detect cars that are further away as in test image 3.

![alt text][image8]

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
This [link](./project_out_frame3.mp4) show the video output with the same exact pipleline developed for a single frame. The result shows jittery detections and some false positives. This [link](./project_output_3_1.mp4) shows the video where I've processed multiple frame at a time to smooth out the result. 


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

In cell 34 and cell 35 shows the code for processing multiple frames at a time. The code is the same as that for processing a single frame except for storing detections from previous 15 frames. These detections are combined and added to the heatmap and a threshold of yy seemed to produce the best result. 

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

1. In this project I learned a great deal about image gradiants, color and spatial features. It helped me develop some intuition about tradditional computer vision techniques. 

2. There is a great deal of tradeoff with increasing the accuracy and reducing execution speed. Increasing accuracy also tended to increase the relative number of misclassifications. This can be mitigated by considering multiple frames but this leads to not being able to detect far away cars and tracking fast moving cars.

3. My pipeline performs poorly when the cars are far away. Mis-identifies oncoming cars in the lane to the left.

4. The pipeline can be improved by increasing the training data and the type of training data. X-dimension maybe used for heatmap and thresholding if your own position is known with respect to traffic. A threshold changing with position may help.

5. A deep-learning approach with CNNs may perform the best.
