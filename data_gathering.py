import cv2
import time
import numpy as np
from gpiozero import Button
# from picamera2 import Picamera2
import h5py

# Note, to make tensorflow lite work on bullfrog, have to use python -m pip install --upgrade tflite-support==0.4.3
from tflite_support.task import core, processor, vision
import utils

models = "efficientdet_lite0.tflite"
# Parameters
# Where is the camera? usually this works, but sometimes you have to find it with v4l2-ctl --list-devices
webcam = "/dev/video0"
# Threading for TF
num_threads = 4
# Display parameters
dispW = 1280  # 1280  # 640 # 
dispH = 720  # 720  # 480  #
# File to save pictures to
pictures_file = "pictures.hdf5"
dset_name = "pics"
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
button.when_pressed = add_picture
pictures = []
# Find the actual resolution, since it forcably changes it
ret, image = cam.read()
dispH = len(image)
dispW = len(image[0])

# Detecting if Boson goes off camera
def edge_detection(x, y, w, h):
    return (min(max(x, 0), max(dispW - w - 1, 0)),
            min(max(y, 0), max(dispH - h - 1, 0)),
            w, h)

# Loop setup stuff
take_pic_flag = False
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


    # Tensorflow
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    im_tensor = vision.TensorImage.create_from_array(image_rgb)
    det_objs = detector.detect(im_tensor)
    # Do something with the data
    # Find the dog and photograph him
    if take_pic_flag:
        dog_bounds = []
        for det_obj in det_objs.detections:
            label = det_obj.categories[0].category_name
            if label == "dog" or label == "cat":
                square_side = max(det_obj.bounding_box.width,  det_obj.bounding_box.height)
                dog_bounds = list(edge_detection(det_obj.bounding_box.origin_x,  det_obj.bounding_box.origin_y,
                                                 square_side, square_side))
                break
        if len(dog_bounds):
            printv(dog_bounds)
            dog_area = cv2.resize(image[dog_bounds[1]:dog_bounds[1]+dog_bounds[3],
                                        dog_bounds[0]:dog_bounds[0]+dog_bounds[2]], (pic_dim, pic_dim))
            # cv2.imshow("Mask", small)  # Draw just the found object of interest for tuning
            image_gray = cv2.cvtColor(dog_area, cv2.COLOR_BGR2GRAY)
            timer = time.time()
            take_pic_flag = False
            cv2.imshow("picture", image_gray)
            time.sleep(0.5)
            another_pic = cv2.waitKey(2500)
            if another_pic == ord(" "):
                take_pic_flag = True
            cv2.destroyWindow("picture")
            if take_pic_flag:
                printv("dog noted")
                if len(pictures):
                    pictures = np.append(pictures, [image_gray], axis=0)
                else:
                    pictures = np.array([image_gray], dtype=np.uint8)
            else:
                printv("dog rejected")
        else:
            printv("no dog detected, no image")
        take_pic_flag = False
        printv(len(pictures))
    image_det = utils.visualize(image, det_objs)


    # Add frame rate
    timer2 = time.time()
    dt = timer2 - timer
    fps = 0.9 * fps + 0.1/ dt
    cv2.putText(image, str(int(np.round(fps))) + " FPS. Button takes a picture, again confirms",
                font_pos, cv2.FONT_HERSHEY_SIMPLEX, font_size, 
                font_color, font_weight)
    timer = timer2

    # Show the image
    cv2.imshow("Camera", image)
    # Exit and saving
    keyHit = cv2.waitKey(1)
    if keyHit == ord(" "):
        take_pic_flag = True
    if keyHit == ord("s"):
        printv("Saving")
        if not len(pictures):
            printv("Error, no pictures")
        else: 
            f = h5py.File(pictures_file, "a")
            if dset_name not in f.keys():
                printv(pictures.shape)
                f.create_dataset(dset_name, data=pictures, chunks=True, 
                                 maxshape=(None,pic_dim,pic_dim))
            else:
                dset = f[dset_name]
                dset.resize(dset.shape[0] + len(pictures), axis=0)
                dset[-len(pictures):] = pictures
            del f
    if keyHit == ord("q") or keyHit == ord("s"):
        leave_loop()
cv2.destroyAllWindows()
