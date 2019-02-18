# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 11:01:16 2019

@author: KK
"""
import cv2
import pytesseract
#This is class to perform OCR operations on a card
class CardOCR:
    def __init__(self,tesseractPath=None):
        self.__image=None
        self.BestString=None
        print(tesseractPath)
        pytesseract.pytesseract.tesseract_cmd = tesseractPath

        
    def Scan(self):
        cam = cv2.VideoCapture(0)
    
        cv2.namedWindow("test")
        
        img_counter = 0
        ret, frame = cam.read()
        frame=cv2.flip(frame,1)
        max_y,max_x=len(frame),len(frame[0])
        #x1,y1=int(0.05*max_x),int(0.05*max_y)
        P1=int(0.03*max_x),int(0.03*max_y)
        P2=int(0.97*max_x),int(11/17*(0.97*max_x-0.03*max_x))
        cv2.rectangle(frame,(P1),(P2),(0, 255, 0), 2)
        cv2.imshow("test", frame)
        while True:
            ret, frame = cam.read()
            frame=cv2.flip(frame,1)
            max_y,max_x=len(frame),len(frame[0])
            cv2.rectangle(frame,(P1),(P2),(0, 255, 0), 2)
            cv2.imshow("test", frame)
            if not ret:
                break
            k = cv2.waitKey(1)
        
            if k%256 == 27:
                # ESC pressed
                print("Escape hit, closing...")
                if self.__image is None:
                    print("No image was captured")
                
                break
            elif k%256 == 32:
                # SPACE pressed
                img_name = "I_{}.png".format(img_counter)
                frame=cv2.flip(frame,1)
                self.__image=frame[P1[1]:P2[1],P1[0]:P2[0]]
                cv2.imwrite(img_name, self.__image)
                print("{} written!".format(img_name))
                img_counter += 1
        
        cam.release()
        
        cv2.destroyAllWindows()
        del cam
        #return img

    def display(self):
        
        if self.__image is None:
            print("Call the Scan Function and capture image before calling display()")
        else:
            cv2.imshow('Image',self.__image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
    def __variance_of_laplacian(self):
    	# compute the Laplacian of the image and then return the focus
    	# measure, which is simply the variance of the Laplacian
    	return cv2.Laplacian(self.__image, cv2.CV_64F).var()

    
    def isBlur(self):
        val=self.__variance_of_laplacian()
        print("Blurry Variance:",val)
        blurthresh=600
        if val<blurthresh:
            return True
        else:
            return False
        
    def OCR(self):
        img_gray = cv2.cvtColor(self.__image, cv2.COLOR_BGR2GRAY)
        
        outputs=[]
        '''manual thresholding'''
        #Bigger number more black, Smaller number more white. Start at 70 till you get a good image
        for thresh in range(70,90,5):
            im_bw = cv2.threshold(img_gray, thresh, 255, cv2.THRESH_BINARY)[1]
            #display(im_bw)
            outputs.append(pytesseract.image_to_string(im_bw,lang="eng"))
            print("----String, Manual Threshold: ",outputs[-1])
        
        self.BestString=self.__chooseBestString(outputs)
        
        self.CardNumber1=self.__parse_1(self.BestString)
        print("Parsed Card Number (Simple): ",self.CardNumber1)
        
        self.CardNumber2=self.__parse_2(self.BestString)
        print("Parsed Card Number (Simple): ",self.CardNumber2)
    
    def __chooseBestString(self, Strings):
        strlen=[]
        for op in Strings:
            strlen.append(sum(c.isdigit() for c in op))
        
        return Strings[strlen.index(max(strlen))]
    
    def __parse_1(self,Bstr):
        card_no=''
        for i in range(0,len(Bstr)):
            if Bstr[i].isnumeric():
                if len(Bstr)>i+19:
                    card_no=Bstr[i:i+19]
                    #if '\n' in card_no:
                    #    card_no=''
                    #else:
                    break
                else:
                    card_no=Bstr[i:-1]
        return card_no
    
    def __parse_2(self,Bstr):
        parts=Bstr.split('\n')
        part=[]
        for p in parts:
            temp=p.split(' ')
            for t in temp:
                part.append(t)
            
        card_num=[]
        for s in part:
            if (s.isnumeric() and len(s)==4):
                card_num.append(s)
                
        return card_num
    
    def Help(self):                    
        print("Functions available are: \n\
              Scan() to open camera and capture image \n\
              isBlur() to test whether image is blurred or not. \n\
              \tReturn True if blur. \n\
              \tReturns False if not blur. \n\
              OCR() to perform OCR on the captured image. It returns parsed card number. \n\
              display() to display the image saved in Scan() ")
        
    