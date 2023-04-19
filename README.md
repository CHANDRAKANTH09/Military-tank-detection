# Military-tank-detection

An autonomous navigation by a military tank towards its target such as another military tank or vehicles, based on the detection, locking-on and following the movements of the target is a much sought after application. We will simulate the war field using cars. A car mounted with NVidia GPU board and HD camera, shall detect other cars and locks-on to one of the nearest cars and follows it.
Technologies : IOT Programming, Deep Learning.

# yolov4-deepsort
Object tracking implemented with YOLOv4, DeepSort, and TensorFlow. YOLOv4 is a state of the art algorithm that uses deep convolutional neural networks to perform object detections. We can take the output of YOLOv4 feed these object detections into Deep SORT (Simple Online and Realtime Tracking with a Deep Association Metric) in order to create a highly accurate object tracker.

### Deep Sort:
DeepSORT can be defined as the tracking algorithm which tracks objects not only based on the velocity and motion of the object but also the appearance of the object.The network is trained on a large-scale person re-identification dataset making it suitable for tracking context. To train the deep association metric model in the DeepSORT cosine metric learning approach is used. According to DeepSORT’s paper, “The cosine distance considers appearance information that is particularly useful to recover identities after long-term occlusions when motion is less discriminative.” That means cosine distance is a metric that helps the model recover identities in case of long-term occlusion and motion estimation also fails. Using these simple things can make the tracker even more powerful and accurate.
![deepsort](https://user-images.githubusercontent.com/95843188/220253013-b9d3e35a-e2ca-40ad-9ccc-7cb4c320f629.jpg)


## Demo of Object Tracker on Tanks
https://user-images.githubusercontent.com/95843188/217168596-433d7547-8cb8-4baf-b139-0045abedac72.mp4\


## Getting Started
To get started, install the proper dependencies Pip.

### Pip
(TensorFlow 2 packages require a pip version >19.0.)
```bash
# TensorFlow CPU
pip install -r requirements.txt

# TensorFlow GPU
pip install -r requirements-gpu.txt
```

## Downloading Official YOLOv4 Pre-trained Weights
Our object tracker uses YOLOv4 to make the object detections, which deep sort then uses to track. There exists an official pre-trained YOLOv4 object detector model that is able to detect 80 classes. For easy demo purposes we will use the pre-trained weights for our tracker.
Download pre-trained yolov4.weights file: https://drive.google.com/open?id=1cewMfusmPjYWbrnuJRuKhPMwRe_b9PaT



If you want to use yolov4-tiny.weights, a smaller model that is faster at running detections but less accurate.

## Running the Tracker with YOLOv4
To implement the object tracking using YOLOv4, first we convert the .weights into the corresponding TensorFlow model which will be saved to a checkpoints folder. Then all we need to do is run the object_tracker.py script to run our object tracker with YOLOv4, DeepSort and TensorFlow.
```bash
# Convert darknet weights to tensorflow model
python save_model.py --model yolov4 

# Run yolov4 deep sort object tracker on video
python object_tracker.py --video ./data/video/test.mp4 --output ./outputs/demo.avi --model yolov4

# Run yolov4 deep sort object tracker on webcam (set video flag to 0)
python object_tracker.py --video 0 --output ./outputs/webcam.avi --model yolov4
```
The output flag allows you to save the resulting video of the object tracker running so that you can view it again later. Video will be saved to the path that you set. (outputs folder is where it will be if you run the above command!)

If you want to run yolov3 set the model flag to ``--model yolov3``, upload the yolov3.weights to the 'data' folder and adjust the weights flag in above commands. (see all the available command line flags and descriptions of them in a below section)


## Resulting Video
As mentioned above, the resulting video will save to wherever you set the ``--output`` command line flag path to. I always set it to save to the 'outputs' folder. You can also change the type of video saved by adjusting the ``--output_format`` flag, by default it is set to AVI codec which is XVID.




## Filter Classes that are Tracked by Object Tracker
By default the code is setup to track all 80 or so classes from the coco dataset, which is what the pre-trained YOLOv4 model is trained on. However, you can easily adjust a few lines of code in order to track any 1 or combination of the 80 classes. It is super easy to filter only the ``person`` class or only the ``car`` class which are most common.

To filter a custom selection of classes all you need to do is comment out line 159 and uncomment out line 162 of [object_tracker.py](https://github.com/theAIGuysCode/yolov4-deepsort/blob/master/object_tracker.py) Within the list ``allowed_classes`` just add whichever classes you want the tracker to track. The classes can be any of the 80 that the model is trained on, see which classes you can track in the file [data/classes/coco.names](https://github.com/theAIGuysCode/yolov4-deepsort/blob/master/data/classes/coco.names)




### References 
  * [yolov3](https://github.com/heartkilla/yolo-v3)
  * [Deep SORT](https://learnopencv.com/understanding-multiple-object-tracking-using-deepsort/)
  * [Yolo paper](https://arxiv.org/abs/1506.02640)
