import cv2
import numpy as np
import time
import math
import copy
import serial

from simple_pid import PID

yP = 0.007
yI = 0.0
yD = 0
check = 0

lock_target = False

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

def nothing(x):
    pass
def get_roundess(contour, area):
    perimeter = cv2.arcLength(contour, True)
    compactness = perimeter**2/(4*math.pi*area)
    roundess = 1/compactness
    return roundess

# cap = cv2.VideoCapture("/home/jonah/Desktop/calibration.avi")

cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
    
while not cap.isOpened():
    # cap = cv2.VideoCapture("/home/jonah/Desktop/calibration.avi")
    cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)

print("Cap Opened")
frameWidth = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
frameHeight = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(frameWidth, frameHeight)

largest_area = 1000
contIndex = -1

cv2.namedWindow("Frame")

cv2.createTrackbar("low H", "Frame", 0, 180, nothing)
cv2.createTrackbar("High H", "Frame", 176, 180, nothing)
cv2.createTrackbar("low S", "Frame", 0, 255, nothing)
cv2.createTrackbar("High S", "Frame", 255, 255, nothing)
cv2.createTrackbar("low V", "Frame", 0, 255, nothing)
cv2.createTrackbar("High V", "Frame", 108, 255, nothing)
cv2.createTrackbar("Gauss", "Frame", 2, 15, nothing)
cv2.createTrackbar("MorphC", "Frame", 3, 15, nothing)
# cv2.createTrackbar("Frame Count", "Frame", 0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)-1), nothing)
# cv2.createTrackbar("Frame Count", "Frame", 0, 643, nothing)
try:
    serialArduino = serial.Serial(port = "/dev/ttyUSB0", baudrate = 9600)
except:
    print("No Serial")

while True:
    try:
        fps = time.time()

        # frame_counter += 1
        # print(frame_counter)
        # print(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)-1))

        # if frame_counter == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        #     frame_counter = 0
        #     cap = cv2.VideoCapture("/home/jonah/Desktop/video_ws/calibration.avi")
        # frameC = cv2.getTrackbarPos("Frame Count", "Frame")
        # cap.set(1, frameC)
        ret, frame = cap.read()
        # frame = cv2.resize(frame,(300,300),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)

        #Gauss
        gauss = cv2.getTrackbarPos("Gauss", "Frame")
        gauss = 2*gauss+1
        frame = cv2.GaussianBlur(frame, (gauss, gauss), cv2.BORDER_DEFAULT)

        datauji = copy.deepcopy(frame)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        low_H = cv2.getTrackbarPos("low H", "Frame")
        high_H = cv2.getTrackbarPos("High H", "Frame")
        low_S = cv2.getTrackbarPos("low S", "Frame")
        high_S = cv2.getTrackbarPos("High S", "Frame")
        low_V = cv2.getTrackbarPos("low V", "Frame")
        high_V = cv2.getTrackbarPos("High V", "Frame")
        
        morph = cv2.getTrackbarPos("Morph", "Frame")

        low_TH = np.array([low_H, low_S, low_V])
        high_TH = np.array([high_H, high_S, high_V])

        frameTH = cv2.inRange(hsv, low_TH, high_TH)
    
        #Close
        morpC = cv2.getTrackbarPos("MorphC", "Frame")
        morpC = 2*morpC+1
        kernel = np.ones((morpC, morpC), np.uint8)
        frameTH = cv2.morphologyEx(frameTH, cv2.MORPH_CLOSE, kernel)

        frameCont = frameTH.copy()

        try:
            _, contours , hie = cv2.findContours(frameCont, mode = cv2.RETR_CCOMP, method = cv2.CHAIN_APPROX_SIMPLE)
        except:
            contours , hie = cv2.findContours(frameCont, mode = cv2.RETR_CCOMP, method = cv2.CHAIN_APPROX_SIMPLE)
        
        hull = []
        hull_index = -1

        if len(contours)>0:
            for j in range(len(contours)):
                area = cv2.contourArea(contours[j])

                if (area>largest_area):
                    largest_area = area
                    contIndex = j
                    hull.append(cv2.convexHull(contours[j], False))
                    hull_index += 1
                    bouding_rect = cv2.boundingRect(hull[hull_index])


        if not contIndex == -1:
            mom = cv2.moments(contours[contIndex])
            m0 = mom["m01"]
            m1 = mom["m10"]
            momArea = mom["m00"]
            # print(momArea)

            CoGX = 130
            centerXX = int(frameWidth/2)
            centerYX = int((frameHeight/2) + CoGX)
            targetXX = int(m1 / momArea)
            targetYX = int(m0 / momArea)
            print(math.sqrt(math.pow((centerXX-targetXX),2)))

            mode1 = targetXX
            mode2 = targetYX
            mode3 = 3
            jarak = 4

            yaw_pid = PID(yP, yI, yD, setpoint=centerXX, output_limits=(-0.75,0.75))
            yaw_axis = yaw_pid(targetXX)
            error = math.sqrt(math.pow((centerXX-targetXX),2))

            if error<20 :
                lock_target = True
                yaw_memory = yaw_axis
                # Beri double check
            else: 
                dataWrite = "{}{},{},{},{},{},{}{}{}".format("*", 90, 0, 30, 90, yaw_axis, 0, "#","\n")

            if lock_target and check >50:
                if error<20:
                    dataWrite = "{}{},{},{},{},{},{}{}{}".format("*", 0, 0, 30, 90, centerXX, 1, "#","\n")
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
            
            cv2.circle(frame, (int(m1/momArea), int(m0/momArea)), 10, (0,0,255))
            cv2.drawContours(frame, contours, contIndex, (0,255,0), 1)
            cv2.drawContours(frame, hull, hull_index, (0,255,0), 5)
            cv2.circle(frame, (targetXX, targetYX), 10, (255,0,0), 3)
            cv2.rectangle(frame, (bouding_rect[0], bouding_rect[1]), (bouding_rect[0]+bouding_rect[2], bouding_rect[1]+bouding_rect[3]), (255,0,0), 1)
            cv2.rectangle(frame, (bouding_rect[0], bouding_rect[1]-int(50)), (bouding_rect[0]+bouding_rect[2], bouding_rect[1]), (255, 255, 255), -1)
            
        cv2.imshow("Frame", frameTH)
        cv2.imshow("Tracking", frame)
        # cv2.imshow("Tracking", datauji)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        largest_area = 0
        contIndex = -1
        time.sleep(0.09)
        # print(1/(time.time()-fps))
    
    except Exception as e:
        print(e)