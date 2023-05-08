#! /usr/bin/env python3

###
# KINOVA (R) KORTEX (TM)
#
# Copyright (c) 2019 Kinova inc. All rights reserved.
#
# This software may be modified and distributed
# under the terms of the BSD 3-Clause license.
#
# Refer to the LICENSE file for details.
#
###

######################################################################################
# To use these examples:                                                             #
#  - Connect to the robot's web page                                                 #
#  - Configure the Vision Color Sensor to 1280x720 resolution                        #
#  - Position the robot so you can easily place objects in front of the Color camera #
#  - Select the Camera view page                                                     #
######################################################################################

import sys
import os
import time

from kortex_api.autogen.client_stubs.VisionConfigClientRpc import VisionConfigClient
from kortex_api.autogen.client_stubs.DeviceManagerClientRpc import DeviceManagerClient

from kortex_api.autogen.messages import DeviceConfig_pb2, Session_pb2, DeviceManager_pb2, VisionConfig_pb2

from Xlib import display

#
# Returns the device identifier of the Vision module, 0 if not found
#
def get_device_id(device_manager):
    vision_device_id = 0
    
    # Getting all device routing information (from DeviceManagerClient service)
    all_devices_info = device_manager.ReadAllDevices()

    vision_handles = [ hd for hd in all_devices_info.device_handle if hd.device_type == DeviceConfig_pb2.VISION ]
    if len(vision_handles) == 0:
        print("Error: there is no vision device registered in the devices info")
    elif len(vision_handles) > 1:
        print("Error: there are more than one vision device registered in the devices info")
    else:
        handle = vision_handles[0]
        vision_device_id = handle.device_identifier
        print("Vision module found, device Id: {0}".format(vision_device_id))

    return vision_device_id

def get_mouse_pos():
    # Get Mouse Cursor Position
    data = display.Display().screen().root.query_pointer()._data
    mx = data["root_x"]
    my = data["root_y"]

    # Convert mouse position to image coordinates
    x = linear_map(321, 1601, 0, 1280, mx)
    y = linear_map(165, 884, 0, 720, my)

    return x, y

def linear_map(x1, x2, y1, y2, x):
        slope = 1.0 * (y2 - y1) / (x2 - x1)
        y = (slope * (x - x1)) +  y1
        return y

def dynamic_focus(vision_config, vision_device_id, x, y):
    sensor_focus_action = VisionConfig_pb2.SensorFocusAction()
    sensor_focus_action.sensor = VisionConfig_pb2.SENSOR_COLOR

    sensor_focus_action.focus_action = VisionConfig_pb2.FOCUSACTION_SET_FOCUS_POINT
    sensor_focus_action.focus_point.x = int(x)
    sensor_focus_action.focus_point.y = int(y)
    vision_config.DoSensorFocusAction(sensor_focus_action, vision_device_id)
    sensor_focus_action.focus_action = VisionConfig_pb2.FOCUSACTION_FOCUS_NOW
    vision_config.DoSensorFocusAction(sensor_focus_action, vision_device_id)

def set_camera_option(vision_config, vision_device_id, option, value):
    sensor = VisionConfig_pb2.SENSOR_COLOR
    option_value = VisionConfig_pb2.OptionValue()
    option_value.sensor = sensor
    options = {'brightness': {'id': 2, 'name': 'OPTION_BRIGHTNESS', 'writable': True, 'min': -4.0, 'max': 4.0, 'step': 1.0, 'default': 0.0},
                'contrast': {'id': 3, 'name': 'OPTION_CONTRAST', 'writable': True, 'min': -4.0, 'max': 4.0, 'step': 1.0, 'default': 0.0},
                'saturation': {'id': 8, 'name': 'OPTION_SATURATION', 'writable': True, 'min': -4.0, 'max': 4.0, 'step': 1.0, 'default': 0.0}}
    option_value.option = options[option]['id']
    if value > options[option]['max'] or value < options[option]['min']:
        print(f"ERROR! Value must be between {options[option]['min']} and {options[option]['max']} at increments of {options[option]['step']}")
    option_value.value = value
    vision_config.SetOptionValue(option_value, vision_device_id)
    

def main():
    # Import the utilities helper module
    # sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    import tools.camtools as utilities

    # Parse arguments
    args = utilities.parseConnectionArguments()
    
    # Create connection to the device and get the router
    with utilities.DeviceConnection.createTcpConnection(args) as router:

        device_manager = DeviceManagerClient(router)
        vision_config = VisionConfigClient(router)

        # example core
        vision_device_id = get_device_id(device_manager)

        if vision_device_id != 0:

            # # Camera options
            # options = ['brightness', 'contrast', 'saturation']

            # # Reset camera options to default
            # for option in options:
            #     set_camera_option(vision_config, vision_device_id, option, 0.0)

            # values = range(-4, 4)
            # for option in options:
            #     for value in values:
            #         input(f"Press ENTER to change {option} to {value}\n")
            #         set_camera_option(vision_config, vision_device_id, option, value)
            #     # Reset values
            #     set_camera_option(vision_config, vision_device_id, option, 0.0)

            input("PRESS ENTER TO Set camera properties...\n")
            set_camera_option(vision_config, vision_device_id, 'brightness', -1.0)
            set_camera_option(vision_config, vision_device_id, 'contrast', -2.0)
            set_camera_option(vision_config, vision_device_id, 'saturation', 0.0)

            input("PRESS ENTER TO START DYNAMIC FOCUS...\n")

            count = 0
            while count < 20:
                count += 1
                x, y = get_mouse_pos()
                print(f"Count #{count} - Mouse Position (u,v): ({int(x)}, {int(y)})")
                dynamic_focus(vision_config, vision_device_id, x, y)
                time.sleep(3.0)

if __name__ == "__main__":
    main()
