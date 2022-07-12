import cv2
import numpy as np
import fingers as fin 
import time
import autopy

# autopy.alert.alert("Hello World!", "1st Test", "default", "cancel")

##############VARIABLES##############
wCam, hCam = 640, 480
wScr, hScr = autopy.screen.size()
frameR = 70 # Frame Reduction
smoothening = 5 # Smoothening factor
plocX, plocY = 0, 0
clocX, clocY = 0, 0
#####################################

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

#1. Find the hand landmarks
#2. get the tip of the index and middle finger (if only single(index) finger is up, it is moving mode, and if both the fingers are up, it is clicking mode.)
#3. Check which fingers are up ✌.
#4. Check whether it is in moving or clicking mode.
#5. If in moving mode, convert our coordinates, (we need to convert as, our web cam shall give us 640x480, but our screen shall be 1920x1080 pixels (full hd 😎))
#6. Smoothen the values.
#7. Move your mouse to those coordinates.
#8. Once in clicking mode, find the distance b/w 2 fingers are short😉.
#9. Display frame rate.

cTime, pTime = 0, 0
detector = fin.handDetector(maxHands=1)

while True :
    success, img = cap.read()
    img = cv2.flip(img, 1)

    img = detector.findHands(img, True)
    lmList, bbox = detector.findPosition(img)

    # Check if there is a hand, i.e, the length of lmlist is not zero.
    if len(lmList) != 0 :
        x1, y1 = lmList[8][1:] # Tip of the index finger
        x2, y2 = lmList[12][1:] # Tip of the middle finger
        # print(x1, y1, x2, y2)

        # When we go sufficiently down, our hand detector will not detect the hand anymore. To fix this ->
        cv2.rectangle(img, (frameR, frameR - 20), (wCam - frameR, hCam - frameR - 25), (0, 0, 0), 2)

        fingers = detector.fingersUp()
        # If only index finger is up, it is moving mode.
        if fingers[1] == 1 and fingers[2] == 0 :
            cv2.circle(img, (x1, y1), 10, (200, 90, 2), cv2.FILLED)
            # Convert coordinates.
            x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
            y3 = np.interp(y1, (frameR - 20, hCam - frameR - 25), (0, hScr))
            clicked = False
            # move the mouse.
            if (x1 >= frameR and x1 <= wCam - frameR) and (y1 >= frameR - 20 and y1 <= hCam - frameR - 25) :
                # Instead of x3 and y3, we can smoothen the values.
                clocX = plocX + (x3 - plocX) / smoothening
                clocY = plocY + (y3 - plocY) / smoothening

                autopy.mouse.move(int(clocX), int(clocY))

                plocX, plocY = clocX, clocY

        # If both the fingers are up, it is clicking mode.
        total = 0
        for i in fingers :
            total += i
        if fingers[1] == 1 and fingers[2] == 1 and total == 2:
            length, img, _ = detector.findDistance(8, 12, img, False)
            # print(length)
            if (length > 15) and (length < 26) :
                # Click condition
                cv2.circle(img, (x1, y1), 10, (200, 90, 2), cv2.FILLED)
                cv2.circle(img, (x2, y2), 10, (200, 90, 2), cv2.FILLED)
                if (not clicked) :
                    autopy.mouse.click(autopy.mouse.Button.LEFT)
                    clicked = True
            else :
                clicked = False
        else:
            clicked = False

    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (250, 40), cv2.FONT_HERSHEY_PLAIN, 2.5, (200, 100, 0), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)