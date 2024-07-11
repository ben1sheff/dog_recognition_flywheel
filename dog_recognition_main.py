import cv2
import torch
import torchvision.transforms as transforms
import h5py
import time
import numpy as np
import torch.optim as optim
import torch.nn as nn
from gpiozero import Button, LED

# Note, to make tensorflow lite work on bullfrog, have to use python -m pip install --upgrade tflite-support==0.4.3
from tflite_support.task import core, processor, vision
import utils

import stance_classifier as sc
# TO DO: this currenty just draws a train vs. validation plot to
# judge epochs of training needed. This is better done with callback
# functions, which I'm sure pyTorch supports
# Parameters
model_param_file = "model_parameters"
pictures_file = "stream_data.hdf5"  # "pictures.hdf5"  # location of the data
confidence_req = 0.6  # model outputs above this are classified as "sitting"

# Parameters
# Where is the camera? usually this works, but sometimes you have to find it with v4l2-ctl --list-devices
webcam = "/dev/video0"
# TF
models = "efficientdet_lite0.tflite"
num_threads = 4
# Display parameters
dispW = 1280  # 1280  # 640 # 
dispH = 720  # 720  # 480  #
pic_dim = 512  # len and wid of picture
# fps label
font_pos = (20, 60)
font = cv2.FONT_HERSHEY_SIMPLEX
font_size = 0.7
font_weight = 2
font_color = (0, 155, 0)
# Do we do print statements?
verbose = True
printv = lambda *args, **kwargs: print(*args, **kwargs) if verbose else None

# Setting up object detection
base_options = core.BaseOptions(file_name=models, use_coral=False, num_threads=num_threads)
detection_options = processor.DetectionOptions(max_results=4, score_threshold = 0.3) # how many objects, how sure to be
options = vision.ObjectDetectorOptions(base_options=base_options, detection_options=detection_options)
detector = vision.ObjectDetector.create_from_options(options)

# Initialize the button
button = Button(18)

# Prepare the camera
cam = cv2.VideoCapture(webcam)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, dispW)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, dispH)
cam.set(cv2.CAP_PROP_FPS, 30)
# Find the actual resolution, since it forcably changes it
ret, image = cam.read()
dispH = len(image)
dispW = len(image[0])

# Load in the data
f = h5py.File(pictures_file, "r")
dset = np.array(f["pics"], dtype="float32")
tags = torch.from_numpy(np.array(f["sitting"], dtype="float32"))
dset = list(zip(dset, tags))

# Prepare the classifier
classifier = sc.StanceClassifier((512, 512))
classifier.load_state_dict(torch.load(model_param_file))
def is_sitting(image, cutoff=confidence_req):
    return classifier(torch.Tensor(image[None,:,:])).item() > cutoff


# Using Tensorflow lite for object detection
def tflite_obj_det(image, color_conv=cv2.COLOR_BGR2RGB):
    ''' Returns an object holding a list of detected objects '''
    image_rgb = cv2.cvtColor(image, color_conv)
    im_tensor = vision.TensorImage.create_from_array(image_rgb)
    return detector.detect(im_tensor)

# Detecting if Boson goes off camera
def edge_detection(x, y, w, h):
    return (min(max(x, 0), max(dispW - w - 1, 0)),
            min(max(y, 0), max(dispH - h - 1, 0)),
            w, h)

# Use detections from tflite_obj_det to make a grayscaled 512x512 dog image
def get_grayscaled_dog(image, det_objects=None, img_size=pic_dim):
    ''' 
    Returns the largest dog/cat in image as a grayscaled array.
    format is img_size x img_size int-array, each 0 to 255
    '''
    if det_objects is None:
        det_objects = tflite_obj_det(image)
    dog_bounds = []
    for det_obj in det_objects.detections:
        label = det_obj.categories[0].category_name
        if label == "dog" or label == "cat":
            square_side = max(det_obj.bounding_box.width,  det_obj.bounding_box.height)
            dog_bounds += [[square_side, list(edge_detection(det_obj.bounding_box.origin_x,  det_obj.bounding_box.origin_y,
                                             square_side, square_side))]]
    if len(dog_bounds):
        dog_bounds = max(dog_bounds)[1]
        # printv(dog_bounds)
        dog_area = cv2.resize(image[dog_bounds[1]:dog_bounds[1]+dog_bounds[3],
                                    dog_bounds[0]:dog_bounds[0]+dog_bounds[2]], (pic_dim, pic_dim))
        return cv2.cvtColor(dog_area, cv2.COLOR_BGR2GRAY)
    return None


# Loop setup stuff
# last_pic_time = time.time()  # Bring back to force a delay between readings, like requiring he sit for a period of time
exit_flag = False
fps = 0  # Start
timer = time.time()
sit_counter = 0
# How to leave the loop
def leave_loop():
    global exit_flag
    exit_flag = True
    cv2.destroyAllWindows()

while not exit_flag:
    ret, image = cam.read()
    # image = picam.capture_array()
    # image = cv2.flip(image, 1)

    # # If it's been 0.5 seconds since the last picture, we can take a new one
    # if (deal_with_pictures_flag 
    #         and (not take_pic_flag 
    #         and time.time() > last_pic_time + 0.5)):
    #     take_pic_flag = True

    # Tensorflow
    det_objs = tflite_obj_det(image)
    # Find the dog and check if he's sitting
    text = ""
    dog_img = get_grayscaled_dog(image, det_objs)
    if dog_img is not None:
        # cv2.imshow("picture", dog_img)
        if is_sitting(dog_img):
            sit_counter += 1
            text = "dog has been sitting " + str(sit_counter) + " frames."
        else:
            text = "dog is not sitting"
            sit_counter = 0
    else:
        text = "no dog detected"
        sit_counter = 0

    # Display TF stuff
    image_det = utils.visualize(image, det_objs)


    # Add frame rate
    timer2 = time.time()
    dt = timer2 - timer
    fps = 0.9 * fps + 0.1/ dt
    cv2.putText(image, str(int(np.round(fps))) + " FPS.     " + text,
                font_pos, cv2.FONT_HERSHEY_SIMPLEX, font_size, 
                font_color, font_weight)
    timer = timer2

    # Show the image
    cv2.imshow("Camera", image)
    # Exit and saving
    keyHit = cv2.waitKey(1)
    # if keyHit == ord(" "):
    #     deal_with_pictures_flag = not deal_with_pictures_flag
    #     if not deal_with_pictures_flag:
    #         cv2.destroyWindow("picture")
    if keyHit == ord("q") or keyHit == ord("s"):
        leave_loop()
cv2.destroyAllWindows()

