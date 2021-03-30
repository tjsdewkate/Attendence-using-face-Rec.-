import cv2
import face_recognition
import numpy as np
import os
from datetime import datetime

# reading the images from the file & make list of images
path = 'attendance'
img = []
ClassNames = []
mylist = os.listdir(path)
print(mylist)

# list of images with appended names
for cl in mylist:
    curimg = cv2.imread(f'{path}/{cl}')
    img.append(curimg)
    ClassNames.append(os.path.splitext(cl)[0])
print(ClassNames)

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def markAttendance(name):
    with open('attend.csv','r+')as f:
        myDataList = f.readline()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')

encodeListknown = findEncodings(img)
print('Encoding Complete')

cap =  cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img,(0,0),None,0.25,0.25)  # we are doing in real time so reducing the size for speading the process
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)

    for encodeFace,faceloc in zip(encodesCurFrame,facesCurFrame):
        matches = face_recognition.compare_faces(encodeListknown,encodeFace)
        faceDis = face_recognition.face_distance(encodeListknown,encodeFace)
        print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches [matchIndex]:
            name = ClassNames[matchIndex].upper()
            #print(name)
            markAttendance(name )

    cv2.imshow('webcam',img)
    cv2.waitKey(1)
