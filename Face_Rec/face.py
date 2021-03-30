import cv2
import face_recognition

# converting the img into RGB (img as BGR)

imgTest = face_recognition.load_image_file('vin2.jpg')
imgTest = cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)
imgTrain = face_recognition.load_image_file('rock.jpg')
imgTrain = cv2.cvtColor(imgTrain,cv2.COLOR_BGR2RGB)

# face detection

faceloc = face_recognition.face_locations(imgTest)[0]
encodevin = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest,(faceloc[3],faceloc[0]),(faceloc[1],faceloc[2]),(255,0,255),2)

faceTest = face_recognition.face_locations(imgTrain)[0]
encodevinTest = face_recognition.face_encodings(imgTrain)[0]
cv2.rectangle(imgTrain,(faceTest[3],faceTest[0]),(faceTest[1],faceTest[2]),(255,0,255),2)

results = face_recognition.compare_faces([encodevin],encodevinTest)

# how similar this iamges are? Distance= lower the distance better the match is
faceDis = face_recognition.face_distance([encodevin],encodevinTest)
print(results,faceDis)

#cv2.putText(imgTrain,f'{results} {round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255).2)

cv2.imshow('vin',imgTest)
cv2.imshow('vin2',imgTrain)
cv2.waitKey(0)