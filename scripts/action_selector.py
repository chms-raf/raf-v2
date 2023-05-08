#!/usr/bin/env python3

import rospy, sys

from raf.msg import RafState

class ActionSelector(object):
    def __init__(self):
        # Params
        self.prev_gui_state = None
        self.gui_state = None
        self.raf_state = RafState()

        # Node cycle rate (in Hz).
        self.loop_rate = rospy.Rate(15)

        # Subscribers
        rospy.Subscriber("/raf_state", RafState, self.state_callback)

        # Publishers
        self.pub = rospy.Publisher('raf_state', RafState, queue_size=10)

    def state_callback(self, msg):
        self.raf_state = msg

    def map2action(self):
        # This function reads the GUI state string and determines the RAF state action
        prev = self.prev_gui_state
        state = self.gui_state

        # Arm camera detections
        # if state == "arm-cam-toggle":
        #     if not self.raf_state.enable_arm_detections or self.raf_state.enable_arm_detections is None:
        #         self.raf_state.enable_arm_detections = True
        #     else:
        #         self.raf_state.enable_arm_detections = False

        if state == "arm-cam-enable":
            self.raf_state.enable_arm_detections = True
        if state == "arm-cam-disable":
            self.raf_state.enable_arm_detections = False
        if state == "visualize-detections-normal":
            self.raf_state.visualize_detections = 'normal'
        if state == "visualize-detections-selection":
            self.raf_state.visualize_detections = 'selection'
        if state == "visualize-detections-disable":
            self.raf_state.visualize_detections = 'disable'

    def publish(self, raf_state_msg):
        self.pub.publish(raf_state_msg)
        self.loop_rate.sleep()


def main():
    """ RAF State Handler """
    rospy.init_node("mdp", anonymous=True)
    run = mdp()

    while not rospy.is_shutdown():
        run.publish(run.raf_state)    

if __name__ == '__main__':
    sys.exit(main())