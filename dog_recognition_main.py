import cv2
import torch
import h5py
import time
import numpy as np
import torch.optim as optim
import torch.nn as nn
from gpiozero import Button, AngularServo, LED
from gpiozero.pins.pigpio import PiGPIOFactory
# To use pigpio, need to run the daemon, with either sudo pigpiod or enable on boot with:
# sudo systemctl enable pigpiod
# sudo systemctl start pigpiod

# Note, to make tensorflow lite work on bullfrog, have to use python -m pip install --upgrade tflite-support==0.4.3
from tflite_support.task import core, processor, vision
import utils
# DETR-based image processing
# # from transformers import DetrImageProcessor, DetrForObjectDetection


import stance_classifier as sc
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
# # DETR object detection
# img_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
# object_detector = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
# Display parameters
dispW = 1280  # 1280  # 640 # 
dispH = 720  # 720  # 480  #
pic_dim = 512  # len and wid of picture

# SERVO
DEFAULT_ANGLE = 10
TRIGGER_ANGLE = 60
MAX_SERVO_ANGLE = 90
SERVO_PIN = 13
DELAY = 2
#Flywheel
FLYWHEEL_PIN = 19
RAMP_TIME = 2
# fps label
font_pos = (20, 60)
font = cv2.FONT_HERSHEY_SIMPLEX
font_size = 0.7
font_weight = 2
font_color = (0, 155, 0)
# Do we do print statements?
verbose = True
printv = lambda *args, **kwargs: print(*args, **kwargs) if verbose else None

# Flywheel
wheel = LED(FLYWHEEL_PIN)
#  servo for swiveling
class Feeder():
    def __init__(self, flywheel=wheel, resting_angle=DEFAULT_ANGLE, trigger_angle=TRIGGER_ANGLE, pin=SERVO_PIN, delay=DELAY):
        ''' Class to handle both the servo feeding the flywheel and when to turn on the flywheel '''
        servo_pin_factory = PiGPIOFactory()
        self.servo = AngularServo(pin,# min_pulse_width = 0.0006, max_pulse_width=0.0023,
                   pin_factory = servo_pin_factory)  # positive angle is anti-clockwise
        self.resting_angle = resting_angle
        self.trigger_angle = trigger_angle
        self.delay = delay
        self.armed = False
        self.flywheel = flywheel
        self.ready_time = time.time()
    def arm(self, flywheel_warmup=RAMP_TIME):
        ''' Start the flywheel and set a delay to ramp it up '''
        self.servo.angle = self.resting_angle
        self.flywheel.on()
        self.ready_time = max(self.ready_time, time.time() + flywheel_warmup)
        self.armed = True
    def trigger(self):
        ''' When to launch the ball, first checks that we have "armed" and readied '''
        if self.armed and time.time() > self.ready_time:
            self.servo.angle = self.trigger_angle
            self.ready_time = max(self.ready_time, time.time() + self.delay)
            self.armed = False
            return True
        return False
    def update(self):
        ''' 
        Call this each loop iteration so we can set delays and act on them without parallelization overhead.
        Currently it just checks if we're either ready to reset after launching or ready after the ramp up
        time after arming
        '''
        if self.ready_time > 0 and time.time() > self.ready_time:
            self.servo.angle = self.resting_angle
            self.ready_time = -1
            if not self.armed:
                self.flywheel.off()
    def cleanup(self):
        ''' reset to our starting setup '''
        self.flywheel.off()
        self.servo.angle = self.resting_angle
feeder = Feeder()

# Setting up object detection
base_options = core.BaseOptions(file_name=models, use_coral=False, num_threads=num_threads)
detection_options = processor.DetectionOptions(max_results=4, score_threshold = 0.3) # how many objects, how sure to be
options = vision.ObjectDetectorOptions(base_options=base_options, detection_options=detection_options)
detector = vision.ObjectDetector.create_from_options(options)
# # openCV approach
# cv_obj_det = cv2.dnn.readNet(model="frozen_inference_graph.pb", config="ssd_mobilenet_v2_coco_2018_03_29.pbtxt.txt", framework="TensorFlow")

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
    with torch.no_grad():
        return classifier(torch.Tensor(image[None,:,:])).item() > cutoff


# function for object detection
class SingleDetection():
    def __init__(self, label, score, box):
        self.label = label
        self.score = score
        self.box = box
class DetectedObjects():
    ''' 
    class to read in object classification to ease
    uniformity between tflite and pytorch
    '''
    def __init__(self, labels, scores, boxes):
        n = len(labels)
        self.labels = labels
        self.scores = scores
        self.boxes = boxes
        self.size = len(self.labels)
    def conv_xyxy_xywh(self):
        ''' Since convert from xmin, ymin, xmax, ymax to x, y, width, height '''
        self.boxes = np.array([
            [x, y, xmax - x, ymax - y] for x, y, xmax, ymax in self.boxes])
    def __iter__(self):
        self.i = 0
        return self
    def __next__(self):
        if self.i >= self.size:
            raise StopIteration
        out = SingleDetection(self.labels[self.i], self.scores[self.i], self.boxes[self.i])
        self.i += 1
        return out
def obj_detection(image, color_conv=cv2.COLOR_BGR2RGB):
    ''' Returns an object holding a list of detected objects '''    
    image_rgb = cv2.cvtColor(image, color_conv)
    # # OpenCV dNN object detection
    # blob = cv2.dnn.blobFromImage(image=image, size=(300, 300), mean=(123, 117, 104))
    # cv_obj_det.setInput(blob)
    # output = cv_obj_det.forward

    # # DETR processing, turns out to be way too slow
    # processed_img = img_processor(image_rgb)
    # processed_img['pixel_values'] = torch.Tensor(np.array(processed_img['pixel_values']))
    # processed_img['pixel_mask'] = torch.Tensor(np.array(processed_img['pixel_mask']))
    # with torch.no_grad():
    #     detr_out = object_detector(**processed_img)
    #     target_sizes = (len(image), len(image[0]))
    #     objects = img_processor.post_process_object_detection(detr_out, target_sizes=target_sizes, threshold=0.9)
    # scores = objects['scores'].tolist()
    # labels = objects['labels'].tolist()
    # boxes = objects['boxes'].tolist()
    # objects = DetectedObjects(labels, scores, boxes)
    # objects.conv_xyxy_xywh()
    # return objects

    # TF Lite Processing
    im_tensor = vision.TensorImage.create_from_array(image_rgb)
    tflite_res = detector.detect(im_tensor)
    n = len(tflite_res.detections)
    labels = []
    scores = []
    boxes = []
    for i in range(n):
        detection = tflite_res.detections[i]
        labels += [detection.categories[0].category_name]
        scores += [detection.categories[0].score]
        boxes += [[detection.bounding_box.origin_x,  detection.bounding_box.origin_y,
                 detection.bounding_box.width, detection.bounding_box.height]]
    objects = DetectedObjects(labels, scores, boxes)
    return objects

# Detecting if Boson goes off camera
def edge_detection(x, y, w, h):
    return (min(max(x, 0), max(dispW - w - 1, 0)),
            min(max(y, 0), max(dispH - h - 1, 0)),
            w, h)
def add_labels(det_objs, image, color=(0, 0, 255)):
    for det_obj in det_objs:
        x, y, w, h = det_obj.box
        cv2.rectangle(image, (x, y), (x + w, y + h), color)
        cv2.putText(image, det_obj.label,
                (x, y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                color, 2)

# Use detections from tflite_obj_det to make a grayscaled 512x512 dog image
def get_grayscaled_dog(image, det_objects=None, img_size=pic_dim):
    ''' 
    Returns the largest dog/cat in image as a grayscaled array.
    format is img_size x img_size int-array, each 0 to 255
    '''
    if det_objects is None:
        det_objects = obj_detection(image)
    dog_bounds = []
    for det_obj in det_objects:
        label = det_obj.label
        if label == "dog" or label == "cat":
            square_side = max(det_obj.box[2],  det_obj.box[3])
            dog_bounds += [[square_side, list(edge_detection(det_obj.box[0], det_obj.box[1],
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
motor_flag = False
timer = time.time()
sit_counter = 0
# How to leave the loop
def leave_loop():
    global exit_flag
    exit_flag = True
    cv2.destroyAllWindows()
    feeder.cleanup()

while not exit_flag:
    ret, image = cam.read()
    # image = picam.capture_array()
    # image = cv2.flip(image, 1)

    # # If it's been 0.5 seconds since the last picture, we can take a new one
    # if (deal_with_pictures_flag 
    #         and (not take_pic_flag 
    #         and time.time() > last_pic_time + 0.5)):
    #     take_pic_flag = True

    # object detection
    det_objs = obj_detection(image)
    add_labels(det_objs, image)
    # Find the dog and check if he's sitting
    text = ""
    dog_img = get_grayscaled_dog(image, det_objs)
    if dog_img is not None:
        # cv2.imshow("picture", dog_img)
        if is_sitting(dog_img):
            sit_counter += 1
            text = "dog has been sitting " + str(sit_counter) + " frames."
            if sit_counter == 5:
                feeder.trigger()
        else:
            text = "dog is not sitting"
            sit_counter = 0
            # if motor_flag:
            #     motor.off()

    else:
        text = "no dog detected"
        sit_counter = 0
        # if motor_flag:
        #     motor.off()
    # Display TF stuff
    # image_det = utils.visualize(image, det_objs)
    feeder.update()

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
    if keyHit == ord("a"):
        feeder.arm()
    if keyHit == ord("q") or keyHit == ord("s"):
        leave_loop()
cv2.destroyAllWindows()

