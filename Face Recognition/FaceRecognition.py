# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 13:40:50 2019

@author: dell
"""
import cv2
import numpy as np
import keras.backend as K
from keras.models import load_model
import pytesseract

class FaceRecognition:

    def euclidean_distance(self,vects):
        x, y = vects
        sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
        return K.sqrt(K.maximum(sum_square, K.epsilon()))
    
    
    def eucl_dist_output_shape(self,shapes):
        shape1, shape2 = shapes
        return (shape1[0], 1)
    
    
    def contrastive_loss(self,y_true, y_pred):
        '''Contrastive loss from Hadsell-et-al.'06
        http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
        '''
        margin = 1
        square_pred = K.square(y_pred)
        margin_square = K.square(K.maximum(margin - y_pred, 0))
        return K.mean(y_true * square_pred + (1 - y_true) * margin_square)
    
    
    
    def __init__(self,tesseractPath=None):
       
        print(tesseractPath)
        pytesseract.pytesseract.tesseract_cmd = tesseractPath
    
    
    def initcaffemodel(self):
        print("[INFO] loading face detector...")
        protoPath = "E:/face_detector/deploy.prototxt" #os.path.sep.join(["E:\!Kunal\ML\Liveness\face_detector", "deploy.prototxt"])
        modelPath = "E:/face_detector/res10_300x300_ssd_iter_140000.caffemodel" #os.path.sep.join(["E:\!Kunal\ML\Liveness\face_detector", "res10_300x300_ssd_iter_140000.caffemodel"])
        net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
        return net
    
    def capture(self):
        
        if not 'net' in locals():
            # myVar exists.
            net=self.initcaffemodel()
        
        cap = cv2.VideoCapture(0)
        
        
        while(True):
            # Capture frame-by-frame
            ret, frame = cap.read()
            
            invGamma = 1.0 / 2
            table = np.array([((i / 255.0) ** invGamma) * 255
            for i in np.arange(0, 256)]).astype("uint8")
            
            # apply gamma correction using the lookup table
            frame=cv2.LUT(frame, table)
            
            (h, w) = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        
        	# pass the blob through the network and obtain the detections and
        	# predictions
            net.setInput(blob)
            detections = net.forward()
            
            for i in range(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.6:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
        
        			# ensure the detected bounding box does fall outside the
        			# dimensions of the frame
                    startX = int(max(0, startX-10))
                    startY = int(max(0, startY-15))
                    endX = int(min(w, 10+endX))
                    endY = int(min(h, 15+endY))
                    
                    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
                    
                    face = frame[startY:endY, startX:endX]
                    #face cv2.resize()
                    #cv2.imwrite("Kunal")
                    break;
                    
        
            # Display the resulting frame
            cv2.imshow('frame',frame)
            if cv2.waitKey(1) & 0xFF == ord(' '):
                break
        
        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()
        return face
    
    
    #face1=capture()
    #face2=capture()
    
    
    
    def display(self,img,msg="Image"):
        cv2.imshow(msg,img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def help():
        print("Functions available are: \n\
                  Capture() to open camera and capture image \n\
                  isBlur() to test whether image is blurred or not. \n\
                  \tReturn True if blur. \n\
                  \tReturns False if not blur. \n\
                  faceverify() to perform OCR on the captured image. It returns parsed card number and parsed expiry date and valid variable which tells whether parsed card is a valid card number or not. \n\
                  display() to display the image saved in Scan() \n")
            
    #face_resize2=cv2.resize(face,(92,112)) 
    
    #cv2.imwrite("KK2.jpg",face_resize2)
    
    def __variance_of_laplacian(self,face):
        # compute the Laplacian of the image and then return the focus
        # measure, which is simply the variance of the Laplacian
        return cv2.Laplacian(face, cv2.CV_64F).var()

    
    def isBlur(self,face):
        val=self.__variance_of_laplacian(face)
        print("Blurry Variance:",val)
        blurthresh=600
        if val<blurthresh:
            return True
        else:
            return False
        
        
    def faceverify(self,face2):
        model = load_model('28-04-2019evenlargermodel.h5', custom_objects={'contrastive_loss': self.contrastive_loss})
        
        #face1=cv2.imread("KK1.jpg",1)
        #face2=cv2.imread("KK2.jpg",1)
        
        
        #face=capture()
        #faceresize= cv2.resize(face,(92,112))
        #cv2.imwrite("AK1.jpg",faceresize)
        #face1=cv2.imread('AK1.jpg',0)
        face1= cv2.imread("AK1.jpg",-1)
        #----------------------------
       # face2=self.capture()
        
        f1=cv2.resize(face1,(92,112))
        f2=cv2.resize(face2,(92,112))
        
        f1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
        f2 = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)
        self.display(f1)
        self.display(f2)
        
        f1=np.expand_dims(f1,axis=2)
        f2=np.expand_dims(f2,axis=2)
        #face=np.expand_dims(face,axis=)
        
        f1=f1.astype('float32')
        f1 /= 255
        f2=f2.astype('float32')
        f2 /= 255
        
        
        f1=np.expand_dims(f1,axis=0)
        f2=np.expand_dims(f2,axis=0)
        '''
        face=cv2.resize(face,(92,112))
        face=cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face=np.expand_dims(face,axis=2)
        face=face.astype('float32')
        face/=255
        face=np.expand_dims(face,axis=0)
        '''
        y_pred = model.predict([f1,f2])
        
        
        if y_pred.ravel()<0.2:
            print("It matches")
        else:
            print("It does not match")
