#!/usr/bin/python

import rospy
from sensor_msgs.msg import Image

from video import StereoVision

from cv_bridge import CvBridge

right_image = None
left_image = None

def right_image_callback(data):
  global right_image
  right_image = data

def left_image_callback(data):
  global left_image
  left_image = data

def run():
  global right_image
  global left_image

  # Init node
  rospy.init_node("stereo_vision");

  # Subscribe for two "eyes"
  rospy.Subscriber("/stereo/right/image_raw", Image, right_image_callback)
  rospy.Subscriber("/stereo/left/image_raw", Image, left_image_callback)

  stereo_publisher = rospy.Publisher("/vision/stereo", Image, queue_size=1000)

  # Action
  print "Start processing..."

  bridge = CvBridge()
  worker = StereoVision()

  # rate = rospy.Rate(10)

  while not rospy.is_shutdown():

    if right_image != None and left_image != None:

      cv_right_image = bridge.imgmsg_to_cv2(right_image, desired_encoding="bgr8")

      cv_left_image = bridge.imgmsg_to_cv2(left_image, desired_encoding="bgr8")

      cv_stereo_image = worker.create_stereo_image( cv_right_image, cv_left_image )

      stereo_image = bridge.cv2_to_imgmsg(cv_stereo_image, encoding="bgr8")

      stereo_publisher.publish( stereo_image )

    # rate.sleep()

if __name__ == "__main__":
  run()
