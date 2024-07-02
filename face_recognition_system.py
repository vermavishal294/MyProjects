import cv2
import numpy as np

cap=cv2.VideoCapture(0)
skip=0
face_data=[]

dataset_path='./Data/'
file_name=input("Enter you Name : ")
# Face detection
face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

while True:
    ret,frame=cap.read()

    if ret==False:
        continue

    
    faces=face_cascade.detectMultiScale(frame,1.3,5)
    faces=sorted(faces,key=lambda f:f[2]*f[3])
                                   
    for face in faces[-1:]:
        x,y,w,h=face
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
        
        offset=10
        face_section=frame[y-offset:y+h+offset,x-offset:x+w+offset]
        face_section=cv2.resize(face_section,(100,100))
        
         #Store every 10th face
        skip+=1
        if skip%10==0:
            face_data.append(face_section)
            print(len(face_data))
        
    cv2.imshow("video frame",frame)
    cv2.imshow("Face Section",face_section)
                
    print(faces)
                                   
    

    
    key_pressed=cv2.waitKey(1) & 0xFF

    if key_pressed==ord('q'):
        break

face_data=np.asarray(face_data)
face_data=face_data.reshape((face_data.shape[0],-1))
np.save(dataset_path+file_name +'.npy',face_data)
cap.release()
cv2.destroyAllWindows()