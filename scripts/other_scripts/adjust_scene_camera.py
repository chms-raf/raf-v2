#!/usr/bin/env python

# This allows you to adjust the scene camera static transform incrementally to roughly align arm and scene camera point clouds.

# The resulting transformation must be added manually to the raf_robot.launch launch file.

# Author: Jack Schultz
# Created 10/12/22

from resource import RLIMIT_OFILE
import rospy
import sys
from curtsies import Input
import tf
import tf2_ros
import geometry_msgs.msg
import threading

def thread_job(run):
    # Trigger the kill signal
    with Input(keynames='curses') as input_generator:
        for e in input_generator:
            # Translation
            # X - forward/backward  - 'q'/'e'
            # Y - left/right        - 'a'/'d'
            # Z - up/down           - 'w'/'s'
            if e == 'a':
                print("Camera move left.")
                run.Y += 0.01
            elif e == 'd':
                print("Camera move right.")
                run.Y -= 0.01
            elif e == 'w':
                print("Camera move up.")
                run.Z += 0.01
            elif e == 's':
                print("Camera move down.")
                run.Z -= 0.01
            elif e == 'q':
                print("Camera move forward.")
                run.X += 0.01
            elif e == 'e':
                print("Camera move backward.")
                run.X -= 0.01
            # Rotation
            # Roll  - 'o'/'p'
            # Pitch - 'KEY_UP'/'KEY_DOWN'
            # Yaw   - 'KEY_LEFT'/'KEY_RIGHT'
            elif e == 'KEY_LEFT':
                print("Camera yaw left.")
                run.Roll += 0.01
            elif e == 'KEY_RIGHT':
                print("Camera yaw right.")
                run.Roll -= 0.01
            elif e == 'KEY_UP':
                print("Camera pitch up.")
                run.Pitch -= 0.01
            elif e == 'KEY_DOWN':
                print("Camera pitch down.")
                run.Pitch += 0.01
            elif e == 'o':
                print("Camera roll left.")
                run.Yaw -= 0.01
            elif e == 'p':
                print("Camera roll right.")
                run.Yaw += 0.01
            else:
                print("Invalid input.")

class adjust_scene_camera():
    def __init__(self):
        # Initialize parameters
        self.X = -0.67
        self.Y = -0.54
        self.Z = 0.75
        self.Yaw = 0.05
        self.Pitch = 0.47
        self.Roll = 0.29

    def spinOnce(self):
        r = rospy.Rate(10)
        r.sleep()

def main(run):
    # rospy.init_node("adjust_scene_camera")

    br = tf2_ros.StaticTransformBroadcaster()
    camFix = geometry_msgs.msg.TransformStamped()
    camFix.header.stamp = rospy.Time.now()
    camFix.header.frame_id = "/scene_camera/camera_link"
    camFix.child_frame_id = "/scene_camera_link"
    camFix.transform.translation.x = float(0)
    camFix.transform.translation.y = float(0)
    camFix.transform.translation.z = float(0)
    camFix_quat = tf.transformations.quaternion_from_euler(
    float(0),float(0),float(0))
    camFix.transform.rotation.x = camFix_quat[0]
    camFix.transform.rotation.y = camFix_quat[1]
    camFix.transform.rotation.z = camFix_quat[2]
    camFix.transform.rotation.w = camFix_quat[3]

    
    broadcaster = tf2_ros.StaticTransformBroadcaster()
    cam2robot = geometry_msgs.msg.TransformStamped()
    cam2robot.header.stamp = rospy.Time.now()
    cam2robot.header.frame_id = "/base_link"
    cam2robot.child_frame_id = "/scene_camera/base_link"

    while not rospy.is_shutdown():
        cam2robot.transform.translation.x = float(run.X)
        cam2robot.transform.translation.y = float(run.Y)
        cam2robot.transform.translation.z = float(run.Z)

        quat = tf.transformations.quaternion_from_euler(
        float(run.Yaw),float(run.Pitch),float(run.Roll))
        cam2robot.transform.rotation.x = quat[0]
        cam2robot.transform.rotation.y = quat[1]
        cam2robot.transform.rotation.z = quat[2]
        cam2robot.transform.rotation.w = quat[3]

        br.sendTransform(camFix)
        broadcaster.sendTransform(cam2robot)
        run.spinOnce()

    return 0

if __name__ == '__main__':

    # Initialize ROS Node
    rospy.init_node("adjust_scene_camera", anonymous=True)

    run = adjust_scene_camera()

    # Start keyboard input thread
    e = threading.Thread(target=thread_job, args=(run,))
    e.setDaemon(True) 
    e.start()

    try:
        sys.exit(main(run))    
    finally:
        print('Program Terminated.')