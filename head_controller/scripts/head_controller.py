#!/usr/bin/python

import rospy

from sensor_msgs.msg import JointState
from std_msgs.msg import Float64
from geometry_msgs.msg import Quaternion
from tf.transformations import euler_from_quaternion

import numpy as np

head_mapping = None
head_joints = {}
center_view = None

def min_max(value, min_value, max_value):

  if value > max_value:
    return max_value

  if value < min_value:
    return min_value

  return value

def set_angle(name, angle):
  global head_mapping
  global head_joints

  motors = head_mapping[name]

  for motor in motors:

    value = -angle if "-" in motor else angle
    motor = motor.replace("-", "")

    if motor in head_joints:
      head_joints[motor].publish( Float64( value ) )

def set_orientation( x, y, z ):
  global center_view

  x = min_max(
      x + (np.pi / 2),
      (np.pi / 2) - (np.pi / 4),
      (np.pi / 2) + (np.pi / 4) )

  y = min_max(
      y - (np.pi / 4),
      -(np.pi / 2),
      (np.pi / 4) )

  z = min_max(
      -z,
      -(np.pi / 2) - (np.pi / 4),
      (np.pi / 2) + (np.pi / 4) )

  print x
  print y
  print z
  print "---------------"

  set_angle("base", x)
  set_angle("neck", y)
  set_angle("eye", z)

def orientation_callback(data):
  global center_view

  quat = [ data.x, data.y, data.z, data.w ]

  # Get head angles in Euler
  euler = euler_from_quaternion( quat )

  x = euler[1]
  y = euler[0]
  z = euler[2]

  if center_view == None:
    center_view = x

  x -= center_view

  set_orientation( x, y, z )

def run():
  global head_mapping
  global head_joints

  # Init the Node
  rospy.init_node("view_controller")

  # Read quarternion orientation from the Oculus
  rospy.Subscriber("/oculus/orientation", Quaternion, orientation_callback)

  # Mapping of motors
  head_mapping = rospy.get_param("~head_mapping")

  for joint, motor_list in head_mapping.items():
    for motor in motor_list:
      motor = motor.replace("-", "")
      head_joints[motor] = rospy.Publisher("/%s/command" % motor, Float64, queue_size=10)

  rate = rospy.Rate(1)
  time = 1

  while True:
    # Update to center
    set_orientation(0, 0, 0)

    # Countdown
    time = time - 1

    if time == 0:
      break

    # Sleep a moment
    rate.sleep()

  # Stil here
  rospy.spin()

if __name__ == '__main__':
  run()
