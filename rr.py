import sys
from facenet_pytorch import InceptionResnetV1
sys.path.append('./License_Plate_Detection_Pytorch/LPRNet')
sys.path.append('./License_Plate_Detection_Pytorch/MTCNN')
import torch
from PIL import Image
import cv2
import cvzone
import matplotlib.pyplot as plt
import torchvision
import argparse
import os
import numpy as np
from collections import Counter
from datetime import datetime
from MTCNN import *
from LPRNet_Test import *
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import time
import pytesseract


harcascade = "haarcascade_russian_plate_number.xml"
plate_detector = cv2.CascadeClassifier(harcascade)
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    edges = cv2.Canny(thresh, 30, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    chars = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 100 and area < 500:  # adjust threshold values
            x, y, w, h = cv2.boundingRect(contour)
            chars.append((x, y, w, h))
    
    plate_text = pytesseract.image_to_string(frame, lang='eng', config='--psm 11')
    cv2.imshow("Edges and Contours", frame)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()