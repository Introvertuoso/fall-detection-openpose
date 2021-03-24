# From Python
# It requires OpenCV installed for Python
import sys
import time

import cv2
import os
from sys import platform
import argparse

import numpy as np

critical_speed = 0.09  # m/s
critical_angle = 45  # degrees
critical_ratio = 1.0  # w/h
small_number = 0.000001


# Setting OpenPose parameters
def set_params():
    params = dict()
    params["model_folder"] = "../../../models/"
    params["disable_blending"] = True
    # params["logging_level"] = 3
    # params["output_resolution"] = "-1x-1"
    # params["net_resolution"] = "-1x368"
    # params["model_pose"] = "BODY_25"
    # params["alpha_pose"] = 0.6
    # params["scale_gap"] = 0.3
    # params["scale_number"] = 1
    # params["render_threshold"] = 0.05
    # # If GPU version is built, and multiple GPUs are available, set the ID here
    # params["num_gpu_start"] = 0
    # params["disable_blending"] = False
    # # Ensure you point to the correct path where models are located
    # params["default_model_folder"] = dir_path + "/../../../models/"
    return params


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


def speed(old_center, old_time, new_center, new_time):
    velocity = -1

    # TODO: Tweak this
    return 0.095

    delta_time = new_time - old_time
    velocity = abs(old_center[1] - new_center[1]) / delta_time

    return [velocity]


def angle(head, center):
    return np.arctan(abs( (head[1]-center[1])/(head[0]-center[0]) ))


def ratio(head, r_knee, l_knee):
    max_x = max([head[0], r_knee[0], l_knee[0]])
    min_x = min([head[0], r_knee[0], l_knee[0]])
    max_y = max([head[1], r_knee[1], l_knee[1]])
    min_y = min([head[1], r_knee[1], l_knee[1]])

    return (max_x - min_x) / (max_y - min_y)


def fall_detection_approach_1(old_keypoints, old_time, new_keypoints, new_time):
    if old_keypoints is not None and new_keypoints is not None:
        for old_person, new_person in zip(old_keypoints, new_keypoints):
            if not are_keypoints_valid(new_person) or not are_keypoints_valid(old_person):
                continue

            new_head, new_neck, new_center, new_r_knee, new_l_knee = \
                new_person[0], new_person[1], new_person[8], new_person[10], new_person[13]
            old_head, old_neck, old_center, old_r_knee, old_l_knee = \
                old_person[0], old_person[1], old_person[8], old_person[10], old_person[13]

            old_angle = angle(old_head, old_center)
            old_ratio = ratio(old_head, old_r_knee, old_l_knee)

            curr_speed = speed(old_center, old_time, new_center, new_time)
            curr_angle = angle(new_head, new_center)
            curr_ratio = ratio(new_head, new_r_knee, new_l_knee)

            # and curr_angle < old_angle:
            # and curr_ratio > old_ratio \
            if curr_speed >= critical_speed \
                    and \
                    curr_angle < critical_angle \
                    and \
                    curr_ratio >= critical_ratio:

                print("A Fall was detected...")
                return True
            else:
                return False

    else:
        return False


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
    if stream.isOpened() is False:
        print("Error opening video stream or file")
    font = cv2.FONT_HERSHEY_SIMPLEX
    o_keypoints = None
    o_time = time.time()

    while True:
        ret, img = stream.read()

        # Obtains each frame from the stream and passes it onto the pose estimators
        datum = op.Datum()
        datum.cvInputData = img
        opWrapper.emplaceAndPop(op.VectorDatum([datum]))

        # Outputs the keypoint matrix
        # print("Body keypoints: \n" + str(datum.poseKeypoints))

        n_time = time.time()
        n_keypoints = datum.poseKeypoints
        fall_detection_approach_1(o_keypoints, o_time, n_keypoints, n_time)
        o_keypoints = n_keypoints
        o_time = n_time

        # Outputs the stream with the keypoints highlighted
        cv2.imshow('Fall Detection', datum.cvOutputData)

        # Exits stream windows
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    stream.release()
    cv2.destroyAllWindows()


except Exception as e:
    print(e)
    sys.exit(-1)
