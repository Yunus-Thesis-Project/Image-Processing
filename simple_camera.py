import cv2
from timecontext import Timer

import numpy as np
import os
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
print(tf.__version__)
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
import time
import serial
import math


from object_detection.utils import label_map_util

from object_detection.utils import visualization_utils as vis_util

from simple_pid import PID

yP = 0.007
yI = 0.0
yD = 0
check = 0
yaw_memory = 90

lock_target = False

# path to the frozen graph:
PATH_TO_FROZEN_GRAPH = '/home/ajietb/Skripsi/frozen_inference_graph.pb'

# path to the label map
PATH_TO_LABEL_MAP = '/home/ajietb/Skripsi/object-detection.pbtxt'

# number of classes 
NUM_CLASSES = 1

try:
    serialArduino = serial.Serial(port = "/dev/ttyUSB0", baudrate = 9600)
except:
    print("No Serial")

# gstreamer_pipeline returns a GStreamer pipeline for capturing from the CSI camera
# Defaults to 1280x720 @ 60fps
# Flip the image by setting the flip_method (most common values: 0 and 2)
# display_width and display_height determine the size of the window on the screen


def gstreamer_pipeline(
    capture_width=300,
    capture_height=300,
    display_width=300,
    display_height=300,
    framerate=60,
    flip_method=0,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )


def show_camera():
    global yP, yI, yD, check, yaw_memory, lock_target
    # To flip the image, modify the flip_method parameter (0 and 2 are the most common)
    print(gstreamer_pipeline(flip_method=0))
    cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
    frameWidth = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    frameHeight = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    
    #reads the frozen graph
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    label_map = label_map_util.load_labelmap(PATH_TO_LABEL_MAP)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    i = 0
    # Detection
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            while True:
                # for x in os.listdir():
                fps = time.time()
                # Read frame from camera
                ret, image_np = cap.read()
                # image_np = cv2.resize(image_np, (50, 50), interpolation = cv2.INTER_AREA)

                # path = "D:\\models-master\\train\\"+x
                # image_np = cv2.imread(path)
                # image_np = cv2.resize(image_np, (400, 300), interpolation = cv2.INTER_AREA)
                # print(image_np.shape)
            
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image_np, axis=0)
                # Extract image tensor
                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                # Extract detection boxes
                boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                # Extract detection scores
                scores = detection_graph.get_tensor_by_name('detection_scores:0')
                # Extract detection classes
                classes = detection_graph.get_tensor_by_name('detection_classes:0')
                # Extract number of detections
                num_detections = detection_graph.get_tensor_by_name(
                    'num_detections:0')
                # Actual detection.
                (boxes, scores, classes, num_detections) = sess.run(
                    [boxes, scores, classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})
                
                _index = np.argmax(np.squeeze(scores))
                score = np.squeeze(scores)[_index]
                x1 = np.asarray(300 * np.squeeze(boxes)[_index])[1]
                y1 = np.asarray(300 * np.squeeze(boxes)[_index])[0]
                x2 = np.asarray(300 * np.squeeze(boxes)[_index])[3]
                y2 = np.asarray(300 * np.squeeze(boxes)[_index])[2]
                cat = np.squeeze(classes)[_index]

                # print(score)
                # print(x1,y1,x2,y2)
                # print(cat)

                if score>0.6:
                    cv2.rectangle(image_np, (x1,y1), (x2,y2), (0,255,0), 2)
                    cv2.circle(image_np, (int((x1+x2)/2), int((y1+y2)/2)), 5, (0,255,0), 2)

                    target_X = int((x1+x2)/2)
                    yaw_pid = PID(yP, yI, yD, setpoint=int(frameWidth/2), output_limits=(-0.75,0.75))
                    yaw_axis = yaw_pid(target_X)
                    error = math.sqrt(math.pow((target_X-int(frameWidth/2)),2))

                    if error<20 :
                        lock_target = True
                        yaw_memory = yaw_axis
                    else:
                        dataWrite = "{}{},{},{},{},{},{}{}{}".format("*", 90, 0, 30, 90, yaw_axis, 0, "#","\n")
                    
                    if lock_target and check >50:
                        if error<20:
                            dataWrite = "{}{},{},{},{},{},{}{}{}".format("*", 0, 0, 30, 90, int(frameWidth/2), 1, "#","\n")
                        else:
                            lock_target = False
                            check = 0
                    else:
                        check += 1
                        yaw_memory = yaw_axis
                        dataWrite = "{}{},{},{},{},{},{}{}{}".format("*", 90, 0, 30, 90, yaw_memory, 0, "#","\n")
                    print(check)
                    print(dataWrite)
                    try:
                        serialArduino.write(dataWrite.encode())
                    except:
                        print("No Serial")
                else:
                    dataWrite = "{}{},{},{},{},{},{}{}{}".format("*", 90, 0, 30, 90, 0, 0, "#","\n")  
                    try:
                        serialArduino.write(dataWrite.encode())
                    except:
                        print("No Serial")
                    print("None of Object Detected") 

                # Visualization of the results of a detection.
                # vis_util.visualize_boxes_and_labels_on_image_array(
                # image_np,
                # np.squeeze(boxes),
                # np.squeeze(classes).astype(np.int32),
                # np.squeeze(scores),
                # category_index,
                # use_normalized_coordinates=True,
                # line_thickness=3
                # )
            # Display output
                cv2.imshow('Gun Detection', cv2.resize(image_np, (640, 480)))
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    break
                print(1/(time.time()-fps))
                # time.sleep(0.5)
                # cv2.waitKey(0)
                # break

if __name__ == "__main__":
    show_camera()
