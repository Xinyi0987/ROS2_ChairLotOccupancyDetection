# ROS2_ChairLotOccupancyDetection

## Objective
This repository contains code for the functionality to check chair lot occupancy[^1] status. The purpose of this is for the Year 2 Trimester 1 project, to provide and implement solutions that can help the future of SIT library @ Punggol campus.

Technologies used for Object Detection:
- Tensorflow Lite
- Opencv
- Robot Operating System 2 (ROS2)

Note: This repository is an add on from this [repository](https://github.com/monopolyroku/ROS2_Chair_Detection) to include the codes for checking of chair lot occupancy status.

## How to Use This Code
1. Run the instructions mentioned in the repository mentioned above.
2. Create a workspace on your laptop
3. Git clone ROS2_ChairLotOccupancyDetection repository.
4. Copy the **src** folder into your workspace
5. Inside the directory (**src/chair_detect/chair_detect**), edit `chair_detect_v3.py`, line 266 `default=0` with the camera id that you are using.
6. `colcon build` your workspace
7. $ `ros2 run chair_detect chair_detect`

## Expected Outcome


https://user-images.githubusercontent.com/59643277/225886229-e476ff65-6443-46d1-ac86-252dc6de4a06.mp4



[^1]: Parking Lot for Chairs
