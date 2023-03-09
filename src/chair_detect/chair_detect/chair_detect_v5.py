# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Main script to run the object detection routine."""
import argparse
import sys
import time
import rclpy    
import numpy as np
from rclpy.node import Node
from std_msgs.msg import Int8MultiArray 
from sensor_msgs.msg import Image

import cv2
from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision

_MARGIN = 10  # pixels
_ROW_SIZE = 10  # pixels
_FONT_SIZE = 1
_FONT_THICKNESS = 1
_TEXT_COLOR = (0, 0, 255)  # red
_TEXT_COLOR_GREEN = (0, 255, 0) # green
_FONT = cv2.FONT_HERSHEY_PLAIN

# Creates a list that will contain detected chairs 
detected_chairs = [0, 1, 0, 1]


class ChairDetector(Node):
  
  def __init__(self):
    super().__init__('chair_detector')
    self.publisher_ = self.create_publisher(Int8MultiArray, 'sod_topic', 10)

class ChairLot:
    # line thickness
    thickness = 5
    # For Logitech camera
    # start point: bottom right, bottom left, top left, top right
    startpt = np.array([[374,242],[65,242],[155,120],[355,120]])
    # end point: bottom right, bottom left, top left, top right
    endpt = np.array([[605,485],[312,485],[320,195],[520,190]])
    # text position for status
    txtpos = np.array([[386,260],[77,260],[167,138],[367,138]])
    # chairlot status: 1 for chair in lot, 0 for chair not in lot
    status = [0, 0, 0, 0]
    # first check: chair within chair lot
    initCheck = np.array([[False],[False],[False],[False]])
    # check: chair still within chair lot but chair box flickers sometimes
    check = np.array([[False],[False],[False],[False]])
    # count exist: how long the chair is within lot
    cntNE = np.array([[0],[0],[0],[0]])
    # count not exist: how long the chair is not within lot
    cntE = np.array([[0],[0],[0],[0]])

    def drawBox(self, frame):
        for i in range(4):
            if self.status[i] == True:
                stat = "Status: occupied"
                color = _TEXT_COLOR
                count = "Availabe Count: " + str(self.cntE[i])
            else:
                stat = "Status: available"
                color = _TEXT_COLOR_GREEN
                count = "Occupied Count: " + str(self.cntNE[i])
            frame = cv2.rectangle(frame, self.startpt[i], self.endpt[i], color, self.thickness)
            cv2.putText(frame, stat, self.txtpos[i], _FONT, 0.9, color, 1)
            cv2.putText(frame, count, (self.txtpos[i][0], self.txtpos[i][1]+10), _FONT, 0.9, color, 1)
    
    def checkBox(self, box):
        for i in range(4):
            # chair box startpt1 x and endpt1 x is within the chair lot box x section
            if(box.origin_x >= self.startpt[i][0] and (box.origin_x + box.width) <= self.endpt[i][0]):
                # chair box startpt y and endpt y is within the chair lot box y section
                if(box.origin_y >= self.startpt[i][1] and (box.origin_y + box.height) <= self.endpt[i][1]):
                    self.cntE[i] = 0
                    if(self.check[i] == False):
                       self.check[i] = 1
                    elif(self.check[i] == True and self.cntNE[i] < 20):
                       self.cntNE[i] += 1
                    elif(self.cntNE[i] == 20):
                      #  self.cntNE[i] = 0
                       self.status[i] = 1
                else:
                    # wait for approximately 5 seconds before it change status to available
                    if(self.cntE[i] == 100):
                        self.status[i] = 0
                        self.initCheck[i] = False
                        self.check[i] = False
                        # self.cntE[i] = 0
                        self.cntNE[i] = 0
                    else:
                        self.cntE[i] += 1
            else:
                # wait for approximately 5 seconds before it change status to available
                if(self.cntE[i] == 100):
                    self.status[i] = 0
                    self.initCheck[i] = False
                    self.check[i] = False
                    # self.cntE[i] = 0
                    self.cntNE[i] = 0
                else:
                    self.cntE[i] += 1

      

def visualize(
    image: np.ndarray,
    detection_result: processor.DetectionResult
) -> np.ndarray:
    """Draws bounding boxes on the input image and return it.

    Args:
        image: The input RGB image.
        detection_result: The list of all "Detection" entities to be visualize.

    Returns:
        Image with bounding boxes.
    """

    cl = ChairLot()
    cl.drawBox(image)
    
    for detection in detection_result.detections:
        category = detection.categories[0]
        if category.category_name.lower() == 'chair':  # Only draw bounding box for chair detections
            # Draw bounding_box
            bbox = detection.bounding_box
            start_point = bbox.origin_x, bbox.origin_y
            end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
            cv2.rectangle(image, start_point, end_point, _TEXT_COLOR, 3)

            # Draw label and score
            category_name = category.category_name
            probability = round(category.score, 2)
            result_text = category_name + ' (' + str(probability) + ')'
            text_location = (_MARGIN + bbox.origin_x,
                             _MARGIN + _ROW_SIZE + bbox.origin_y)
            cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                        _FONT_SIZE, _TEXT_COLOR, _FONT_THICKNESS)
            
            # Add chair's coordinates to the list of detected chairs
            center_x = bbox.origin_x + bbox.width/2
            center_y = bbox.origin_y + bbox.height/2
            detected_chairs.append((center_x, center_y))

            # Check whether chair bounding box is within chair lot bounding box
            cl.checkBox(bbox)

    return image




def run(model: str, camera_id: int, width: int, height: int, num_threads: int,
        enable_edgetpu: bool) -> None:
  """Continuously run inference on images acquired from the camera.

  Args:
    model: Name of the TFLite object detection model.
    camera_id: The camera id to be passed to OpenCV.
    width: The width of the frame captured from the camera.
    height: The height of the frame captured from the camera.
    num_threads: The number of CPU threads to run the model.
    enable_edgetpu: True/False whether the model is a EdgeTPU model.
  """

  # Variables to calculate FPS
  counter, fps = 0, 0
  start_time = time.time()

  # Start capturing video input from the camera
  cap = cv2.VideoCapture(camera_id)
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

  # Visualization parameters
  row_size = 20  # pixels
  left_margin = 24  # pixels
  text_color = (0, 0, 255)  # red
  font_size = 1
  font_thickness = 1
  fps_avg_frame_count = 10

  # Initialize the object detection model
  base_options = core.BaseOptions(
      file_name=model, use_coral=enable_edgetpu, num_threads=num_threads)
  detection_options = processor.DetectionOptions(
      max_results=4, score_threshold=0.5)
  options = vision.ObjectDetectorOptions(
      base_options=base_options, detection_options=detection_options)
  detector = vision.ObjectDetector.create_from_options(options)
    
  rclpy.init()

  # Continuously capture images from the camera and run inference
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      sys.exit(
          'ERROR: Unable to read from webcam. Please verify your webcam settings.'
      )

    counter += 1
    # image = cv2.flip(image, 1)

    # Convert the image from BGR to RGB as required by the TFLite model.
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Create a TensorImage object from the RGB image.
    input_tensor = vision.TensorImage.create_from_array(rgb_image)

    # Run object detection estimation using the model.
    detection_result = detector.detect(input_tensor)

    # Draw keypoints and edges on input image
    image = visualize(image, detection_result)

    # Calculate the FPS
    if counter % fps_avg_frame_count == 0:
      end_time = time.time()
      fps = fps_avg_frame_count / (end_time - start_time)
      start_time = time.time()

    # Show the FPS
    fps_text = 'FPS = {:.1f}'.format(fps)
    text_location = (left_margin, row_size)
    cv2.putText(image, fps_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                font_size, text_color, font_thickness)
    
    # chair_detector = ChairDetector()
    point_msg = Int8MultiArray()
    point_msg.data = ChairLot().status
    chair_detector = ChairDetector().publisher_.publish(point_msg)

    # Stop the program if the ESC key is pressed.
    if cv2.waitKey(1) == 27:
      chair_detector.destroy_node()
      rclpy.shutdown()
      break
    cv2.imshow('object_detector', image)

  cap.release()
  cv2.destroyAllWindows()



def main():
  # Sets arguments for parsing
  parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
    '--model',
    help='Path of the object detection model.',
    required=False,
    default='/home/xinyi/testOccupancyP4/detectchair/src/chair_detect/chair_detect/efficientdet_lite0.tflite')
  parser.add_argument(
    '--cameraId', help='Id of camera.', required=False, type=int, default=2)
  parser.add_argument(
    '--frameWidth',
    help='Width of frame to capture from camera.',
    required=False,
    type=int,
    default=640)
  parser.add_argument(
    '--frameHeight',
    help='Height of frame to capture from camera.',
    required=False,
    type=int,
    default=480)
  parser.add_argument(
    '--numThreads',
    help='Number of CPU threads to run the model.',
    required=False,
    type=int,
    default=4)
  parser.add_argument(
    '--enableEdgeTPU',
    help='Whether to run the model on EdgeTPU.',
    action='store_true',
    required=False,
    default=False)
  args = parser.parse_args()
  run(args.model, int(args.cameraId), args.frameWidth, args.frameHeight,
      int(args.numThreads), bool(args.enableEdgeTPU))
  # # Initialization of ROS2 node
  # rclpy.init()
  # node = ChairDetector()
  # rclpy.spin(node)
  # # Destroys ROS2 node
  # node.destroy_node()
  # rclpy.shutdown()



if __name__ == '__main__':
  main()
