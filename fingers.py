from unittest import result
import cv2
import mediapipe as mp
import time # To check the frame-rate.
import math

class handDetector():
    def __init__(self, mode = False, maxHands = 2):
        self.mode = mode
        self.maxHands = maxHands

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands)
        self.mpDraw = mp.solutions.drawing_utils

        self.color_hand_connection = self.mpDraw.DrawingSpec()
        self.color_hand_connection.color = (57, 255, 20)
        self.color_hand_connection.thickness = 1

        self.color_Lms = self.mpDraw.DrawingSpec()
        self.color_Lms.color = (0, 0, 255)
        self.color_Lms.circle_radius = 2

    def findHands(self, img, draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(imgRGB)
        if draw:
            if results.multi_hand_landmarks :
                for handLms in results.multi_hand_landmarks :
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS, self.color_Lms , self.color_hand_connection)
        return img

    def findPosition(self, img, handNo = 0, draw = True):
        xList = []
        yList = []
        bbox = []
        
        self.lmList = []
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(imgRGB)

        if results.multi_hand_landmarks :
            myHand = results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                xList.append(cx)
                yList.append(cy)
                self.lmList.append([id, cx, cy])
            
            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = [xmin, ymin, xmax, ymax]

            if draw :
                cv2.rectangle(img, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20), (0, 255, 0), 2)

        return self.lmList, bbox

    def fingersUp(self) :
        fingers = []
        tipIds = [4, 8, 12, 16, 20] # Thumb, index, middle, ring, pinky (tips)
        # For thumb.
        if (self.lmList[5][1] < self.lmList[17][1] and self.lmList[4][1] < self.lmList[3][1]):
            fingers.append(1)
        elif (self.lmList[5][1] > self.lmList[17][1] and self.lmList[4][1] > self.lmList[3][1]):
            fingers.append(1)
        else:
            fingers.append(0)

        # For fingers. (index, middle, ring, pinky)
        for id in range(1, 5):
            if (self.lmList[tipIds[id]][2] < self.lmList[tipIds[id]-2][2]):
                fingers.append(1) # If fingers are open
            else:
                fingers.append(0) # If fingers are closed
        return fingers

    def findDistance(self, p1, p2, img, draw=True, r=15, t=3):
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), r, (255, 0, 255), cv2.FILLED)
        length = math.hypot(x2 - x1, y2 - y1)
        return length, img, [x1, y1, x2, y2, cx, cy]

def main():
    pTime = 0
    cTime = 0

    cap = cv2.VideoCapture(0)

    while True :
        success, img = cap.read()

        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime

        detector = handDetector()
        img = detector.findHands(img, True)

        cv2.putText(img, f"FPS : {(int(fps))}", (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 100, 0), 2)

        cv2.imshow("Image", img)
        cv2.waitKey(1)
    

if __name__ == "__main__" :
    main()