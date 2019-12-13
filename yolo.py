import numpy as np
import cv2
y0=0
y1=232
y2=259
y3=290
y4=355
y5=416
m1=0.753
m2=1.60
m3=3
ctx = 208
x11=24
x12=48
x13=97
x21=40
x22=48
x23=97
x24=203
x31=149


def velocity(enc):
    if enc >5600:
        return 1
    elif enc >4100 and enc <= 5600:
        return 2
    elif enc > 3020 and enc<=4100:
        return 3
    elif enc > 2700 and enc<=3020:
        return 4
    elif enc > 2400 and enc<=2700:
        return 5
    elif enc > 2200 and enc<=2400:
        return 6
    elif enc > 2055 and enc<=2000:
        return 7
    elif enc > 1900 and enc<=2055:
        return 8
def site_assign(x,y):  
    if y <=y1 and y>y0:
        return 1
    elif y <=y2 and y>y1:
        dist_y= y-y1
        max_lx = ctx - (x21 + m3 * dist_y)
        max_rx = ctx + x21 + m3 * dist_y
        if x <= max_lx:
            return 2
        elif x > max_lx and x<max_rx :
            return 3
        elif x >= max_rx:
            return 4
    elif y <=y3 and y>y2:
        dist_y= y-y2
        max_lx1 = ctx - (x11 + m1 * dist_y)
        max_lx2 = ctx - (x22 + m2 * dist_y)
        max_rx1 = ctx + x11 + m1 * dist_y
        max_rx2 = ctx + x22 + m2 * dist_y
        if x <= max_lx2 :
            return 5
        elif x > max_lx2  and x <= max_lx1:
            return 6
        elif x > max_lx1 and x<= ctx:
            return 7
        elif x > ctx and x<=max_rx1:
            return 8
        elif x > max_rx1 and x <= max_rx2:
            return 9
        elif x > max_rx2 and x <= 416:
            return 10
    elif y <=y4 and y>y3:
        dist_y = y - y3
        max_lx1=ctx-(x12 + m1 * dist_y)
        max_lx2=ctx-(x23 + m2 * dist_y)
        max_rx1=ctx+x12+m1*dist_y
        max_rx2=ctx+x23+m2*dist_y
        if x<=max_lx2 :
            return 11
        elif x> max_lx2 and x<=max_lx1:
            return 12
        elif x> max_lx1 and x<=ctx:
            return 13
        elif x> ctx and x <=max_rx1:
            return 14
        elif x>max_rx1 and x<=max_rx2:
            return 15
        elif x>max_rx2 and x<=416:
            return 16
    elif y<=y5 and y>y4:
        dist_y = y - y4
        max_lx1=ctx-(x13 + m1 * dist_y)
        max_rx1=ctx+x13+m1*dist_y
        if x<=max_lx1 :
            return 17
        elif x>max_lx1 and x<=ctx:
            return 18
        elif x>ctx and x<= max_rx1:
            return 19
        elif x>max_rx1 and x<=416:
            return 20
confidenceThreshold = 0.5
NMSThreshold = 0.3

modelConfiguration = 'yolov3.cfg'
modelWeights = 'yolo-tiny.weights'

labelsPath = 'classes.names'
labels = open(labelsPath).read().strip().split('\n')

np.random.seed(10)
COLORS = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")

net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)

outputLayer = net.getLayerNames()
outputLayer = [outputLayer[i[0] - 1] for i in net.getUnconnectedOutLayers()]

video_capture = cv2.VideoCapture(0)

(W, H) = (None, None)

obj = 0
centerX = 0
centerY = 0
size = 0
while True:
    ret, frame = video_capture.read()
    frame = cv2.flip(frame, 1)
    frame=cv2.resize(frame, (416, 416), interpolation=cv2.INTER_CUBIC)
    if W is None or H is None:
        (H,W) = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB = True, crop = False)
    net.setInput(blob)
    layersOutputs = net.forward(outputLayer)

    boxes = []
    confidences = []
    classIDs = []

    for output in layersOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > confidenceThreshold:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY,  width, height) = box.astype('int')
                x = int(centerX - (width/2))
                y = int(centerY - (height/2))

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    #Apply Non Maxima Suppression
    detectionNMS = cv2.dnn.NMSBoxes(boxes, confidences, confidenceThreshold, NMSThreshold)
    if(len(detectionNMS) > 0):
        mat = []
        #여기에 port.readline을 split하여 행렬 하나 추가 선언
        #arr_serial = []
        for i in detectionNMS.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            color = [int(c) for c in COLORS[classIDs[i]]] # 생략가능
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2) # 생략가능
            text = '{}: {:.4f}'.format(labels[classIDs[i]], confidences[i]) # 생략가능
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2) # 생략가능
            centerX = (x+w)/2
            centerY = (2*y+h)/2
            arr=[classIDs[i], centerY,site_assign(centerX,centerY)]
            #arr.extend(arr_serial)
            mat.append(arr)
        print(mat)
    cv2.line(frame,(0,362),(196,232),(0,0,255),1)
    cv2.line(frame,(200,232),(63,416),(255,0,0),1)
    cv2.line(frame,(208,232),(208,416),(255,0,0),1)
    cv2.line(frame,(216,232),(353,416),(255,0,0),1)
    cv2.line(frame,(220,232),(416,362),(0,0,255),1)
    
    cv2.line(frame,(0,232),(416,232),(255,0,0),1)
    cv2.line(frame,(184,259),(232,259),(255,0,0),1)
    cv2.line(frame,(160,290),(256,290),(255,0,0),1)
    cv2.line(frame,(111,355),(305,355),(255,0,0),1)
    cv2.line(frame,(63,416),(353,416),(255,0,0),2)
    cv2.imshow('Output', frame)
    if(cv2.waitKey(1) & 0xFF == ord('q')):
        break

#Finally when video capture is over, release the video capture and destroyAllWindows
video_capture.release()
cv2.destroyAllWindows()
