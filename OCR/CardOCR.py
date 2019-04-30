# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 11:01:16 2019

@author: KK
"""
import cv2
import pytesseract
import re
#This is class to perform OCR operations on a card
class CardOCR:
    def __init__(self,tesseractPath=None):
        self.__image=None
        self.BestString=None
        print(tesseractPath)
        pytesseract.pytesseract.tesseract_cmd = tesseractPath


    def ReadImg(self,url):
        self.__image=cv2.imread(url,-1)
        
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

    def display(self,string="Image"):
        
        if self.__image is None:
            print("Call the Scan Function and capture image before calling display()")
        else:
            cv2.imshow(string,self.__image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
    def display_img(self,img,string="Image"):
        cv2.imshow(string,img)
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
        reg=r"(\d)|(/)|(\n)|( )"
        outputs=[]
        '''manual thresholding'''
        #Bigger number more black, Smaller number more white. Start at 70 till you get a good image
        for thresh in range(70,105,15):
            im_bw = cv2.threshold(img_gray, thresh, 255, cv2.THRESH_BINARY)[1]
            self.display_img(im_bw)
            tempstr=pytesseract.image_to_string(im_bw,lang="eng")
            temp=""
            for i in range(0,len(tempstr)):
                if re.match(reg, tempstr[i]):
                    temp=temp+tempstr[i]
            outputs.append(temp)
            print("----String, Manual Threshold: ",outputs[-1])
        
        #print("[INFO] performing Canny edge detection...")
        #blurred = cv2.GaussianBlur(img_gray, (5, 5), 1)
        canny = cv2.Canny(img_gray, 30, 150)
        tempstr=pytesseract.image_to_string(canny,lang="eng")
        temp=""
        for i in range(0,len(tempstr)):
            if re.match(reg, tempstr[i]):
                temp=temp+tempstr[i]
        outputs.append(temp)
        print("----String, Canny edge detection: ",outputs[-1])
        self.BestString=self.__chooseBestString(outputs)
        print("Best String: ",self.BestString)
        self.CardNumber1=self.__parse_card_no(self.BestString)
        #print("Parsed Card Number (1): ",self.CardNumber1)
        self.display_img(canny)
        self.ExpiryDate=self.__parse_expiry_no(self.BestString)
        
        return self.CardNumber1,self.ExpiryDate,self.__luhn(self.CardNumber1)
        #self.CardNumber2=self.__parse_2(self.BestString)
        #print("Parsed Card Number (2): ",''.join([ f'{x} ' for x in self.CardNumber2 ]))
    
    '''
    def __chooseBestString(self, Strings):
        strlen=[]
        for op in Strings:
            strlen.append(sum(c.isdigit() for c in op))
        
        return Strings[strlen.index(max(strlen))]
    '''
    def __chooseBestString(self, Strings):
        scores=[]
        for op in Strings:
            score=0
            card_no=self.__parse_card_no(op)
            if self.__luhn(card_no):
                score+=10
            reg=r"(/)"
            for i in range(0,len(op)):
                if re.match(reg, op[i]):
                    score+=3
            scores.append(sum(c.isdigit() for c in op)+score)
        
        return Strings[scores.index(max(scores))]
    
    
    
    def __parse_card_no(self,Bstr):
        card_no=''
        parts=Bstr.split("\n")
        strlen=[]
        for op in parts:
            strlen.append(len(op))
        
        temp=parts[strlen.index(max(strlen))]
        
        reg=r"(\d)"
        card_no=""
        for i in range(0,len(temp)):
            if re.match(reg, temp[i]):
                card_no=card_no+temp [i]
        
        return card_no
    
    def __parse_expiry_no(self,Bstr):
        
        reg=r"(/)"
        flag=False
        for i in range(len(Bstr)-1,0,-1):
            if re.match(reg,Bstr[i]):
                flag=True
                break;
        
        if flag:
            if i-2>=0 and i+3<=len(Bstr):
                expiry=Bstr[i-2:i+3]
            else:
                expiry=""
            
        return expiry
    
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
    
    def __luhn(self,card_no):
        
        if card_no=="":
            return False
        
        sum = 0
        for i,c in enumerate(card_no):
            num = (2-(i % 2)) * int(c)
            sum += int(num/10) + (num % 10)
    	#print sum
        return ((sum % 10) == 0)
	
    
    def Help(self):                    
        print("Functions available are: \n\
              Scan() to open camera and capture image \n\
              isBlur() to test whether image is blurred or not. \n\
              \tReturn True if blur. \n\
              \tReturns False if not blur. \n\
              OCR() to perform OCR on the captured image. It returns parsed card number and parsed expiry date and valid variable which tells whether parsed card is a valid card number or not. \n\
              display() to display the image saved in Scan() \n\
              readImg() is used to read an image from memory to perform OCR on it.")
        
        