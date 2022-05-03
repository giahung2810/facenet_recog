# chup 200 anh khi da detect = MTCNN
import cv2
import os

from facenet_pytorch import MTCNN
import torch

device =  torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(thresholds= [0.7, 0.7, 0.8] ,keep_all=True, device = device)

cap = cv2.VideoCapture(0) 
# cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)   
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
name = input("Enter your Name:")
sampleNum = 1
path= 'data/image'
userNames=os.listdir(path)
if name not in userNames:
    os.makedirs('data/image/'+name)
while (True):
    ret, frame = cap.read()
    boxes, _ = mtcnn.detect(frame)
    if boxes is not None:
        for box in boxes:
            bbox = list(map(int,box.tolist()))
            try:
                cv2.imwrite('data/image/'+ name +'/pic.'  + str(sampleNum) + '.jpg',frame[bbox[1]:bbox[3],bbox[0]:bbox[2]])
                cv2.rectangle(frame,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(0,255,0),2)
                print("Chup anh thu " + str(sampleNum))
                sampleNum += 1
            except Exception as e:
                print('Co loi!')
    
    cv2.imshow('frame',frame)
    cv2.waitKey(100)
    if sampleNum > 200:                 # thoat neu du 200 anh
        break
cap.release()
cv2.destroyAllWindows()