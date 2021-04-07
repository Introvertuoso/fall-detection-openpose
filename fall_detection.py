# From Python
# It requires OpenCV installed for Python
import sys
import time

import cv2
import os
from sys import platform
import argparse

import numpy as np

# TODO: Make a trailer where 2 fight 1 shoots other falls and gun is abandoned with a fire somewhere
# TODO: Multiple people lead to wrong center keypoint association
#           ~get_corresponding_old_keypoints() should fix this (not ultimately)
# TODO: Include the FPS in the threshold to go up for low threshold
#           ~Nevermind this may be due to different resolution and not FPS (Investigate)
#           ~As such, we need to normalize the distance and go from there
#           ~multiplying speed_value_tweak with input_rate/output_rate
#           ~this way better systems are compensated less as apposed to slower systems (ideally 1:1)
#           ~meaning <30ms to process the frame with 30fps cam and half that for 60fps cam and so on...
# TODO: There should be a state machine of some sort to wait for the subject to get up
#           ~we keep track of: centers(to not mix up) + fall_time_stamp
# TODO: Implement the 2nd approach

# TODO: I need to know the FRAME RATE the cam is recording at (PARAMETER FOR PROGRAM)

cam_frame_rate = 30  # My phone's (should be input as a program parameter)
critical_speed = 0.09  # m/s (0.09 originally)
critical_angle = 45  # degrees
critical_ratio = 1.0  # w/h
small_number = 0.000001
average_human_height = 166  # m
torso_to_height_ratio = 38.34 / 100
speed_value_tweak = 7  # Speed values Tweak
meter_value_tweak = 2  # Displacement in meters normalization
height_value_tweak = 100  # Height in pixels normalization
critical_speed = critical_speed * speed_value_tweak


# Setting OpenPose parameters-
def set_params():
    params = dict()
    params["model_folder"] = "../../../models/"
    params["disable_blending"] = True
    params["net_resolution"] = "-1x256"
    # params["render_threshold"] = 0.5
    # params["logging_level"] = 3
    # params["output_resolution"] = "-1x-1"
    # params["alpha_pose"] = 0.6
    # params["scale_gap"] = 0.3
    # params["scale_number"] = 1
    return params


def get_corresponding_old_keypoints(person, old_keypoints):
    # Ideally minDistance should be equal to len of screen diagonal (W x H)
    # So that should be input as well or set statically
    result = None
    minDistance = (1080 * 1920) + 1
    for ok in old_keypoints:
        distance = euclidean_distance(ok[8], person[8])
        if distance < minDistance:
            minDistance = distance
            result = ok

    return result


def euclidean_distance(neck, center):
    point1 = np.array((neck[0], neck[1]))
    point2 = np.array((center[0], center[1]))
    val = ((((center[0] - neck[0] )**2) + (( center[1] - neck[1] )**2) )**0.5)
    return val


# Based on the measurements of Vitrivius later continued by DaVinci
def height_in_pixels_estimator(neck, center):
    val = euclidean_distance(neck, center) / torso_to_height_ratio
    # val = val / height_value_tweak  # To tone down the numbers
    return val


def pixel_to_meter_converter(pixel_displacement, neck, center):
    val = (average_human_height * pixel_displacement) / height_in_pixels_estimator(neck, center)
    # val = val / meter_value_tweak  # To tone down the numbers
    return val


# Checks if the pose estimators are able to pinpoint the joint/keypoint in question.
# This will later also check the confidence of the estimation and filter based on it as well.
def are_keypoints_valid(person):
    head_x, head_y = person[0][0], person[0][1]
    neck_x, neck_y = person[1][0], person[1][1]
    center_x, center_y = person[8][0], person[8][1]
    r_knee_x, r_knee_y = person[10][0], person[10][1]
    l_knee_x, l_knee_y = person[13][0], person[13][1]
    if 0 not in (head_x, neck_x, center_x, r_knee_x, l_knee_x,
                 head_y, neck_y, center_y, r_knee_y, l_knee_y):
        # print("A set of keypoints was found...")
        return True
    else:
        return False


def speed(old_center, old_time, new_center, new_time, neck):
    velocity = -1

    delta_time = new_time - old_time
    displacement_in_pixels = abs(old_center[1] - new_center[1])
    displacement_in_meters = pixel_to_meter_converter(displacement_in_pixels, neck, new_center)
    velocity = displacement_in_meters / delta_time
    # velocity = velocity / speed_value_tweak  # To tone down the numbers
    return velocity


def angle(head, center):
    return np.arctan(abs((head[1] - center[1]) / (head[0] - center[0])))


def ratio(head, r_knee, l_knee):
    max_x = max([head[0], r_knee[0], l_knee[0]])
    min_x = min([head[0], r_knee[0], l_knee[0]])
    max_y = max([head[1], r_knee[1], l_knee[1]])
    min_y = min([head[1], r_knee[1], l_knee[1]])

    return (max_x - min_x) / (max_y - min_y)


def fall_detection_approach_1(old_keypoints, old_time, new_keypoints, new_time):
    if not are_keypoints_valid(new_keypoints) or not are_keypoints_valid(old_keypoints):
        return False

    new_head, new_neck, new_center, new_r_knee, new_l_knee = \
        new_person[0], new_person[1], new_person[8], new_person[10], new_person[13]
    old_head, old_neck, old_center, old_r_knee, old_l_knee = \
        old_person[0], old_person[1], old_person[8], old_person[10], old_person[13]

    old_angle = angle(old_head, old_center)
    old_ratio = ratio(old_head, old_r_knee, old_l_knee)

    curr_speed = speed(old_center, old_time, new_center, new_time, new_neck)
    curr_angle = angle(new_head, new_center)
    curr_ratio = ratio(new_head, new_r_knee, new_l_knee)

    if curr_speed >= critical_speed \
            and \
            curr_angle < critical_angle and curr_angle <= old_angle \
            and \
            curr_ratio >= critical_ratio and curr_ratio >= old_ratio:
        print(curr_speed)
        return True
    else:
        return False


def fall_detection_approach_2():



try:
    # Import Openpose (Windows/Ubuntu/OSX)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    try:
        # Windows Import
        if platform == "win32":
            # Change these variables to point to the correct folder (Release/x64 etc.)
            sys.path.append(dir_path + '/../../python/openpose/Release');
            os.environ['PATH'] = os.environ['PATH'] + ';' + dir_path + '/../../x64/Release;' + dir_path + '/../../bin;'
            import pyopenpose as op
        else:
            # Change these variables to point to the correct folder (Release/x64 etc.)
            sys.path.append('../../python');
            # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the
            # OpenPose/python module from there. This will install OpenPose and the python library at your desired
            # installation path. Ensure that this is in your python path in order to use it. sys.path.append(
            # '/usr/local/python')
            from openpose import pyopenpose as op
    except ImportError as e:
        print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this '
              'Python script in the right folder?')
        raise e

    # Flags
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", default="../../../examples/media/COCO_val2014_000000000192.jpg",
                        help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
    args = parser.parse_known_args()

    # Custom Params (refer to include/openpose/flags.hpp for more parameters)
    params = set_params()

    # Add others in path?
    for i in range(0, len(args[1])):
        curr_item = args[1][i]
        if i != len(args[1]) - 1:
            next_item = args[1][i + 1]
        else:
            next_item = "1"
        if "--" in curr_item and "--" in next_item:
            key = curr_item.replace('-', '')
            if key not in params:  params[key] = "1"
        elif "--" in curr_item and "--" not in next_item:
            key = curr_item.replace('-', '')
            if key not in params: params[key] = next_item

    # Construct it from system arguments
    # op.init_argv(args[1])
    # oppython = op.OpenposePython()

    # Starting OpenPose
    # opWrapper = op.WrapperPython(op.ThreadManagerMode.Synchronous)
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    # opWrapper.execute()
    opWrapper.start()

    stream = cv2.VideoCapture(0)
    # stream = cv2.VideoCapture("..\\media\\F01_Trim.mp4")
    if stream.isOpened() is False:
        print("Error opening video stream or file")
    font = cv2.FONT_HERSHEY_SIMPLEX
    o_keypoints = None
    o_time = time.time()
    prev_frame_time = 0
    new_frame_time = 0

    while True:
        ret, img = stream.read()
        if not ret:
            break

        # Obtains each frame from the stream and passes it onto the pose estimators
        datum = op.Datum()
        datum.cvInputData = img
        opWrapper.emplaceAndPop(op.VectorDatum([datum]))

        # Outputs the keypoint matrix
        # print("Body keypoints: \n" + str(datum.poseKeypoints))

        n_time = time.time()
        print(n_time)
        n_keypoints = datum.poseKeypoints
        if o_keypoints is not None and n_keypoints is not None:
            for old_person, new_person in zip(o_keypoints, n_keypoints):
                old_person = get_corresponding_old_keypoints(new_person, o_keypoints)
                if fall_detection_approach_1(old_person, o_time, new_person, n_time):
                    print("A Fall was detected...")
        o_keypoints = n_keypoints
        new_frame_time = n_time

        frame = datum.cvOutputData
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        fps = int(fps)
        fps = str(fps)
        cv2.putText(frame, fps, (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)

        # Outputs the stream with the keypoints highlighted
        cv2.imshow('Fall Detection', frame)

        # Exits stream windows
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    stream.release()
    cv2.destroyAllWindows()


except Exception as e:
    print(e)
    sys.exit(-1)
