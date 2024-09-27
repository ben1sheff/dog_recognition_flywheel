import cv2
import time
import numpy as np
from gpiozero import Button
# from picamera2 import Picamera2
import h5py

# Note, to make tensorflow lite work on bullfrog, have to use python -m pip install --upgrade tflite-support==0.4.3
from tflite_support.task import core, processor, vision
import utils

# Notes on keys: " " toggles if sitting, "z" activates data gathering, "s" saves and quits, "q" just quits
# If using the button, holding it down toggles sitting, as does releasing it (don't mix with space bar)

# Parameters
# Where is the camera? usually this works, but sometimes you have to find it with v4l2-ctl --list-devices
webcam = "/dev/video0"
# TF
models = "efficientdet_lite0.tflite"
num_threads = 4
# Display parameters
dispW = 1280  # 1280  # 640 # 
dispH = 720  # 720  # 480  #
# File to save pictures to
pictures_file = "stream_data.hdf5"
dset_name = "pics"
sit_key = "sitting"
pic_dim = 512  # len and wid of picture saved
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


button = Button(18)
# picam = Picamera2()
# picam.preview_configuration.main.size = (dispW, dispH)
# picam.preview_configuration.main.format = "RGB888"
# picam.preview_configuration.controls.FrameRate = 30
# picam.preview_configuration.align()  # Forces to standard size for speed
# picam.configure("preview")  # Applies the above configurations
# picam.start()

cam = cv2.VideoCapture(webcam)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, dispW)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, dispH)
cam.set(cv2.CAP_PROP_FPS, 30)

# Setting up to take a series of pictures
def add_picture():
    global take_pic_flag
    take_pic_flag = True
def toggle_state():
    global sitting_state
    sitting_state = not sitting_state
button.when_pressed = toggle_state
button.when_released = toggle_state
# Find the actual resolution, since it forcably changes it
ret, image = cam.read()
dispH = len(image)
dispW = len(image[0])

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
    # # Torch processing
    # processed_img = processor(image_rgb)
    # with torch.no_grad():
    #     objects = model(**processed_img)
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
pictures = []
sitting = []
sitting_state = False
deal_with_pictures_flag = False
take_pic_flag = True
last_pic_time = time.time()
exit_flag = False
fps = 0  # Start
timer = time.time()
# How to leave the loop
def leave_loop():
    global exit_flag
    exit_flag = True
    cv2.destroyAllWindows()

while not exit_flag:
    ret, image = cam.read()
    # image = picam.capture_array()
    # image = cv2.flip(image, 1)
    # # Cut us down to 512x512
    # image = image[dispH//2 - pic_dim//2: dispH//2 + pic_dim//2,
    #               dispW//2 - pic_dim//2: dispW//2 + pic_dim//2]

    # If it's been 0.5 seconds since the last picture, we can take a new one
    if (deal_with_pictures_flag 
            and (not take_pic_flag 
            and time.time() > last_pic_time + 0.5)):
        take_pic_flag = True
    # Tensorflow
    det_objs = obj_detection(image)
    # Find the dog and photograph him
    if deal_with_pictures_flag and take_pic_flag :
        # cv2.imshow("picture", dog_img)
        dog_img = get_grayscaled_dog(image, det_objs)
        if dog_img is not None:
            cv2.imshow("picture", dog_img)
            pictures += [dog_img]
            sitting += [sitting_state]
            printv("dog", len(pictures), "noted, this time he is", "" if sitting_state else "not", "sitting")
        else:
            printv("no dog detected, no image")
        take_pic_flag = False
        last_pic_time = time.time()
    # Display TF stuff
    add_labels(det_objs, image)


    # Add frame rate
    timer2 = time.time()
    dt = timer2 - timer
    fps = 0.9 * fps + 0.1/ dt
    text = str(int(np.round(fps))) + " FPS. "
    text += "z: " + ("stop" if deal_with_pictures_flag else "start") + " pictures. "
    text += "Space: mark " +(" is" if sitting_state else "not") +  " sitting. "
    text += "s: save, q: quit"
    cv2.putText(image, text,
                font_pos, cv2.FONT_HERSHEY_SIMPLEX, font_size, 
                font_color, font_weight)
    timer = timer2

    # Show the image
    cv2.imshow("Camera", image)
    # Exit and saving
    keyHit = cv2.waitKey(1)
    if keyHit == ord("z"):
        deal_with_pictures_flag = not deal_with_pictures_flag
        if not deal_with_pictures_flag:
            cv2.destroyWindow("picture")
    if keyHit == ord(" "):
        toggle_state()
        printv("Sitting is", sitting_state)
    if keyHit == ord("s"):
        printv("Saving")
        if not len(pictures):
            printv("Error, no pictures")
        else: 
            f = h5py.File(pictures_file, "a")
            pictures = np.array(pictures, dtype=np.uint8)
            if dset_name not in f.keys():
                printv(pictures.shape)
                f.create_dataset(dset_name, data=pictures, chunks=True, 
                                 maxshape=(None,pic_dim,pic_dim))
                f.create_dataset(sit_key, data=np.array(sitting), chunks=True, 
                                 maxshape=(None,))
            else:
                dset = f[dset_name]
                dset.resize(dset.shape[0] + len(pictures), axis=0)
                dset[-len(pictures):] = pictures
                sit_set = f[sit_key]
                sit_set.resize(sit_set.shape[0] + len(sitting), axis=0)
                sit_set[-len(sitting):] = np.array(sitting)
                printv("We now have", dset.shape[0], "pictures")
            del f
        leave_loop()
    if keyHit == ord("q"):
        leave_loop()
cv2.destroyAllWindows()
