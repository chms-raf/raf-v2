#!/usr/bin/env python3

# This script handles communication between the GUI and the system by:
#   ...listening for commands from the GUI and updating the RAF state
#   ...listening for requests from other nodes
#   ...sending commands to the GUI
#   ...continuously publishing the RAF state for other nodes

import rospy, sys

from raf.msg import RafState
from std_msgs.msg import String

from argparse import Namespace
import tools.camtools as camtools
from kortex_api.autogen.client_stubs.VisionConfigClientRpc import VisionConfigClient
from kortex_api.autogen.client_stubs.DeviceManagerClientRpc import DeviceManagerClient
from kortex_api.autogen.messages import DeviceConfig_pb2, VisionConfig_pb2
import random

class RafStateHandler(object):
    def __init__(self):
        # Params
        self.from_gui = None
        self.request = None
        self.raf_state = RafState()
        # TODO: REMOVE THIS!
        self.raf_state.enable_face_detections = True
        self.raf_state.view = "arm"
        self.raf_state.system_state = 'idle'

        # Node cycle rate (in Hz).
        self.loop_rate = rospy.Rate(15)

        # Subscribers
        rospy.Subscriber("/from_gui", String, self.gui_callback)
        rospy.Subscriber("/raf_state_request", String, self.request_callback)
        rospy.Subscriber("/focus_request", String, self.focus_callback)

        # Publishers
        self.state_pub = rospy.Publisher('raf_state', RafState, queue_size=10)
        self.gui_pub = rospy.Publisher('to_gui', String, queue_size=10)
        self.pub_msg = rospy.Publisher('raf_message', String, queue_size=10)

    def gui_callback(self, msg):
        self.from_gui = msg.data
        self.mapGUI2state()

    def request_callback(self, msg):
        self.request = msg.data
        self.mapRequest2state()

    def focus_callback(self, msg):
        self.request = msg.data
        
        if self.request == 'focus':
            self.focus_camera()

        rospy.sleep(0.5)

        self.gui_pub.publish("disable-focus")
        self.loop_rate.sleep()

    def focus_camera(self):
        args = Namespace()
        args.ip = '137.148.209.35'
        args.password = 'admin'
        args.username = 'admin'

        with camtools.DeviceConnection.createTcpConnection(args) as router:

            self.device_manager = DeviceManagerClient(router)
            self.vision_config = VisionConfigClient(router)

            # example core
            self.vision_device_id = self.get_device_id()

            if self.vision_device_id != 0:
                sensor_focus_action = VisionConfig_pb2.SensorFocusAction()
                sensor_focus_action.sensor = VisionConfig_pb2.SENSOR_COLOR

                u = random.randrange(0, 1280)
                v = random.randrange(0, 720)

                sensor_focus_action.focus_action = VisionConfig_pb2.FOCUSACTION_SET_FOCUS_POINT
                sensor_focus_action.focus_point.x = int(u)
                sensor_focus_action.focus_point.y = int(v)
                self.vision_config.DoSensorFocusAction(sensor_focus_action, self.vision_device_id)
                sensor_focus_action.focus_action = VisionConfig_pb2.FOCUSACTION_FOCUS_NOW
                self.vision_config.DoSensorFocusAction(sensor_focus_action, self.vision_device_id)

    def get_device_id(self):
        self.vision_device_id = 0
    
        # Getting all device routing information (from DeviceManagerClient service)
        all_devices_info = self.device_manager.ReadAllDevices()

        vision_handles = [ hd for hd in all_devices_info.device_handle if hd.device_type == DeviceConfig_pb2.VISION ]
        if len(vision_handles) == 0:
            print("Error: there is no vision device registered in the devices info")
        elif len(vision_handles) > 1:
            print("Error: there are more than one vision device registered in the devices info")
        else:
            handle = vision_handles[0]
            self.vision_device_id = handle.device_identifier
            print("Vision module found, device Id: {0}".format(self.vision_device_id))

        return self.vision_device_id

    def mapGUI2state(self):
        # This function reads the GUI command string and determines the RAF state

        # Reset button. Should probably only be available for debugging
        if self.from_gui == "reset":
            self.raf_state.enable_arm_detections = False
            self.raf_state.enable_scene_detections = False
            self.raf_state.enable_face_detections = False
            self.raf_state.visualize_face_detections = False
            self.raf_state.enable_visual_servoing = False
            self.raf_state.visualize_detections = 'disable'
            self.raf_state.view = 'arm'
            self.raf_state.system_state = 'idle'
            self.gui_pub.publish("disable-detections")
            self.loop_rate.sleep()
            self.gui_pub.publish("disable-selection")
            self.loop_rate.sleep()
            self.gui_pub.publish("disable-visualize-detections")
            self.loop_rate.sleep()
            self.gui_pub.publish("disable-face")
            self.loop_rate.sleep()
            self.gui_pub.publish("disable-visualize-face")
            self.loop_rate.sleep()
            self.gui_pub.publish("disable-visual-servoing")
            self.loop_rate.sleep()

        # Change View between scene cam and arm cam
        if self.from_gui == "view-scene":
            self.raf_state.view = "scene"
            if self.raf_state.enable_arm_detections or self.raf_state.enable_scene_detections:
                self.gui_pub.publish("disable-detections")
                self.loop_rate.sleep()
            if self.raf_state.visualize_detections == 'normal+selection':
                self.gui_pub.publish("disable-visualize-detections")
                self.loop_rate.sleep()
                self.gui_pub.publish("disable-selection")
            if self.raf_state.visualize_detections == 'normal':
                self.gui_pub.publish("disable-visualize-detections")
            if self.raf_state.visualize_detections == 'selection':
                self.gui_pub.publish("disable-selection")
            self.raf_state.enable_arm_detections = False
            self.raf_state.visualize_detections = 'disable'
        if self.from_gui == "view-arm":
            self.raf_state.view = "arm"
            if self.raf_state.enable_arm_detections or self.raf_state.enable_scene_detections:
                self.gui_pub.publish("disable-detections")
                self.loop_rate.sleep()
            if self.raf_state.visualize_detections == 'normal+selection':
                self.gui_pub.publish("disable-visualize-detections")
                self.loop_rate.sleep()
                self.gui_pub.publish("disable-selection")
            if self.raf_state.visualize_detections == 'normal':
                self.gui_pub.publish("disable-visualize-detections")
            if self.raf_state.visualize_detections == 'selection':
                self.gui_pub.publish("disable-selection")
            self.raf_state.enable_scene_detections = False
            self.raf_state.visualize_detections = 'disable'

        # Enable or disable face detections
        if self.from_gui == "enable-face-detections":
            self.raf_state.enable_face_detections = True
        if self.from_gui == "disable-face-detections":
            self.raf_state.enable_face_detections = False

        # Show/Hide face detections
        if self.from_gui == "visualize-face-detections":
            if self.raf_state.system_state == "idle" or self.raf_state.system_state == "action":
                self.raf_state.visualize_face_detections = True
        if self.from_gui == "visualize-face-detections-disable":
            self.raf_state.visualize_face_detections = False

        # Enable of disable visual servoing
        if self.from_gui == "enable-visual-servoing":
            self.raf_state.enable_visual_servoing = True
        if self.from_gui == "disable-visual-servoing":
            self.raf_state.enable_visual_servoing = False

        # Enable or disable detection inference
        if self.from_gui == "enable-detections":
            if self.raf_state.view == "arm":
                self.raf_state.enable_arm_detections = True
                self.raf_state.enable_scene_detections = False
            # If scene view is enabled
            if self.raf_state.view == "scene":
                self.raf_state.enable_scene_detections = True
                self.raf_state.enable_arm_detections = False
        if self.from_gui == "disable-detections":
            self.raf_state.enable_arm_detections = False
            self.raf_state.enable_scene_detections = False

        # Show/Hide normal detections
        if self.from_gui == "visualize-detections-normal":
            # If in the middle of selection, allow detections to be viewed
            if self.raf_state.visualize_detections == 'selection':
                self.raf_state.visualize_detections = 'normal+selection'
            else:
                self.raf_state.visualize_detections = 'normal'
        if self.from_gui == "visualize-detections-normal-disable":
            if self.raf_state.visualize_detections == 'normal+selection':
                self.raf_state.visualize_detections = 'selection'
            else:
                self.raf_state.visualize_detections = 'disable'

        # Show/Hide selection detections
        if self.from_gui == "visualize-detections-selection":
            # If normal detections are shown, display selection on top
            if self.raf_state.visualize_detections == 'normal':
                self.raf_state.visualize_detections = 'normal+selection'
            else:
                self.raf_state.visualize_detections = 'selection'

            # Set the system state to selection
            if self.raf_state.system_state == 'idle' and (self.raf_state.enable_arm_detections or self.raf_state.enable_scene_detections):
                self.raf_state.system_state = 'selection'
                # TODO: Publish an error message if detections are not enabled
        if self.from_gui == "visualize-detections-selection-disable":
            if self.raf_state.visualize_detections == 'normal+selection':
                self.raf_state.visualize_detections = 'normal'
            else:
                self.raf_state.visualize_detections = 'disable'

            # Set the system state to selection
            if self.raf_state.system_state == 'selection':
                self.raf_state.system_state = 'idle'

    def mapRequest2state(self):
        # This function reads the raf state request string and determines the RAF state
        if self.request == 'system-state-action':
            if self.raf_state.system_state == 'selection':
                self.raf_state.system_state = 'action'
            else:
                self.pub_msg.publish("Error(1) - Resetting to Idle!")
                self.raf_state.system_state = 'idle'

        if self.request == 'system-state-selection':
            if self.raf_state.system_state == 'selection':
                self.pub_msg.publish("Selection Canceled.")
                self.raf_state.system_state = 'selection'
            elif self.raf_state.system_state == 'action':
                self.raf_state.system_state = 'selection'

        if self.request == 'system-state-idle':
            if self.raf_state.system_state == 'selection':
                self.pub_msg.publish("Error(2) - Resetting to Idle!")
                self.raf_state.system_state = 'idle'
            elif self.raf_state.system_state == 'action':
                self.raf_state.system_state = 'idle'

        if self.request == 'to-gui-disable-selection':
            self.gui_pub.publish('disable-selection')
            if self.raf_state.visualize_detections == 'normal+selection':
                self.raf_state.visualize_detections = 'normal'
            if self.raf_state.visualize_detections == 'selection':
                self.raf_state.visualize_detections = 'disable'

        if self.request == 'enable-face-detection':
            self.raf_state.enable_arm_detections = False
            self.raf_state.enable_scene_detections = False
            self.raf_state.enable_face_detections = True
        if self.request == 'disable-face-detection':
            self.raf_state.enable_face_detections = False
            self.gui_pub.publish('disable-face-detection')
            self.loop_rate.sleep()

        if self.request == 'enable-visual-servoing':
            self.raf_state.enable_arm_detections = False
            self.raf_state.enable_scene_detections = False
            self.raf_state.enable_visual_servoing = True
        if self.request == 'disable-visual-servoing':
            self.raf_state.enable_visual_servoing = False
            self.gui_pub.publish('disable-visual-servoing')
            self.loop_rate.sleep()

        if self.request == 'enable-arm-detections':
            self.raf_state.enable_arm_detections = True
            self.raf_state.enable_scene_detections = False
            self.raf_state.enable_visual_servoing = False
        if self.request == 'disable-arm-detections':
            self.raf_state.enable_arm_detections = False
            self.gui_pub.publish('disable-detections')
            self.loop_rate.sleep()

        if self.request == 'reset':
            self.raf_state.enable_arm_detections = False
            self.raf_state.enable_scene_detections = False
            self.raf_state.enable_face_detections = False
            self.raf_state.visualize_face_detections = False
            self.raf_state.enable_visual_servoing = False
            self.raf_state.visualize_detections = 'disable'
            self.raf_state.view = 'arm'
            self.raf_state.system_state = 'idle'
            self.gui_pub.publish("disable-detections")
            self.loop_rate.sleep()
            self.gui_pub.publish("disable-selection")
            self.loop_rate.sleep()
            self.gui_pub.publish("disable-visualize-detections")
            self.loop_rate.sleep()
            self.gui_pub.publish("disable-face")
            self.loop_rate.sleep()
            self.gui_pub.publish("disable-visualize-face")
            self.loop_rate.sleep()
            self.gui_pub.publish("disable-visual-servoing")
            self.loop_rate.sleep()
        
    def publish(self, raf_state_msg):
        self.state_pub.publish(raf_state_msg)
        self.loop_rate.sleep()

def main():
    """ RAF State Handler """
    rospy.init_node("raf_state_handler", anonymous=True)
    run = RafStateHandler()

    while not rospy.is_shutdown():
        run.publish(run.raf_state)    

if __name__ == '__main__':
    sys.exit(main())