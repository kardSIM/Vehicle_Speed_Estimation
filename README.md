[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/10OYV7ZvouCXBPJ3oLjEwFBtZnCHuRJgR?usp=sharing
)
# Speed Estimation from Video using YOLOv9 and DeepSORT

This project estimates the speed of objects in a video using YOLOv9 for object detection and DeepSORT for tracking. The estimated speeds are overlaid on the video along with bounding boxes around the detected objects.

## Demo 
<p align="center">
  <img src="demo/car1.gif" alt="demo" width="45%">
  <img src="demo/car2.gif" alt="blur" width="45%">
</p>


## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Acknowledgements](#acknowledgements)
- [References](#references)




## Installation
1. Clone this repository:
  ```
   git clone https://github.com/kardSIM/Vehicle_Speed_Estimation.git
   cd Vehicle_Speed_Estimation
  ```

2. Create new environment:
  - Using Conda
  ```
  conda env create -f conda.yml
  conda activate yolov9-deepsort
  ```

3. Download input video:
  ```
   mkdir content
   wget -P content https://github.com/AarohiSingla/Speed-detection-of-vehicles/raw/main/highway.mp4
   wget -P content https://github.com/AarohiSingla/Speed-detection-of-vehicles/raw/main/highway_mini.mp4
  ```  
4. Run:
  ```
   python object_tracking.py
  ```  




## Usage
1. Prepare the video file:
   - Place the video file in the desired location.
   - Update the `video` argument in the path of the video file.

2. Configure the output video:
   - Update `output` argument in the code to specify the path and filename of the output video file.

3. Transforming Road Image to Bird's Eye View:
   - Update the source points `SOURCE_POLYGONE` and destination points `BIRD_EYE_VIEW` corresponding to the appropriate dimensions of the road.

3. Set the confidence threshold:
   - Adjust the `conf` argument in the code to set the confidence threshold for object detection. Objects with confidence below this threshold will be filtered out.

4. If you want to detect and track certain object on video 
   - Modify the `class_id` argument in the code to specify the class ID for detection. The default value is set to None.

5. If you want to blur certain object while tracking
   - Modify the `bulr_id` argument in the code to specify the class ID for detection. The default value is set to None. 

6. Run the code:
   ```
   # Run object tracking
   python object_tracking.py

   # Run on another file
   python object_tracking.py --video ./data/test.mp4 --output ./output/output.mp4

   # Run car tracking (set class_id to 2 for car)
   python object_tracking.py --class_id 2
   
   # Run tracking on a video with burring certain objects (set blur_id to 7 for truck)
   python object_tracking.py --blur_id 7
   ```

## Acknowledgements
- This code is built upon the YOLOv9 model and the DeepSort algorithm.
- Credits to the authors and contributors of the respective repositories used in this project.

## References
- [Simple Online and Realtime Tracking with a Deep Association Metric](https://arxiv.org/abs/1703.07402)

- [object detection and tracking in videos using the YOLOv9 object detection model and the DeepSORT algorithm](https://github.com/sujanshresstha/YOLOv9_DeepSORT/blob/main/README.md)

