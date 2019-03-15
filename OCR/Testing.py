# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 02:08:32 2019

@author: Kunal
"""
#Functional script CardOCR is the object for this
import pytesseract
#import imageio
import cv2
import numpy as np

#------------------------------------------------------------------------------
def getMaxIdx(values):
    return values.index(max(values))

def Scan():
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
            try:
                img
            except NameError: img=None
            break
        elif k%256 == 32:
            # SPACE pressed
            img_name = "I_{}.png".format(img_counter)
            frame=cv2.flip(frame,1)
            img=frame[P1[1]:P2[1],P1[0]:P2[0]]
            cv2.imwrite(img_name, img)
            print("{} written!".format(img_name))
            img_counter += 1
    
    cam.release()
    
    cv2.destroyAllWindows()
    del cam
    return img

def display(img):
    cv2.imshow('Image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def img_estim(img, thrshld):
    is_light = np.mean(img) > thrshld
    print(np.mean(img))
    return 'light' if is_light else 'dark'

def adjust_gamma(image, gamma=1):

   invGamma = 1.0 / gamma
   table = np.array([((i / 255.0) ** invGamma) * 255
      for i in np.arange(0, 256)]).astype("uint8")

   return cv2.LUT(image, table)


def Gamma_correction(image) :
#     f = cv2.imread(image,1)
    f = image
#     original = cv2.imread(image, 1)
    original = image
    if img_estim(f, 80) == 'dark' :
        print('dark--------')
        gamma = 2.0
        adjusted = adjust_gamma(original, gamma=gamma)
        return adjusted
    else :
        return original
    
def chooseBestString(Strings):
    strlen=[]
    for op in Strings:
        strlen.append(sum(c.isdigit() for c in op))
        
    return Strings[strlen.index(max(strlen))]
    
def parse_1(Bstr):
    card_no=''
    parts=Bstr.split("\n")
    strlen=[]
    for op in parts:
        strlen.append(len(op))        
    card_no=parts[strlen.index(max(strlen))]
    
    return card_no
    
def parse_2(Bstr):
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
def variance_of_laplacian(image):
	# compute the Laplacian of the image and then return the focus
	# measure, which is simply the variance of the Laplacian
	return cv2.Laplacian(image, cv2.CV_64F).var()
#------------------------------------------------------------------------------
pytesseract.pytesseract.tesseract_cmd = r'E:\!Kunal\Tesseract-OCR\tesseract.exe'
import re

'''
img=Scan()
val=variance_of_laplacian(img)
print("Blurry Variance:",val)
blurthresh=600
if val<blurthresh:
    print("Image is blurry, Please Try Again.")
    img=None
'''

img = cv2.imread("credit_card_01.png",-1)
#img=cv2.resize(img,(800,500))
if img is None:
    print("Image not captured/not captured properly. Press Spacebar when window is open to capture image.")
else:
    display(img)
    #img=Gamma_correction(img)
    #display(img)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    reg=r"(\d)|(/)|(\n)|( )"
    outputs=[]
    '''manual thresholding'''
    #Bigger number more black, Smaller number more white. Start at 70 till you get a good image
    for thresh in range(70,100,5):
        im_bw = cv2.threshold(img_gray, thresh, 255, cv2.THRESH_BINARY)[1]
        display(im_bw)
        cv2.imwrite("Card_Image_01_"+str(thresh)+".png",im_bw)
        tempstr=pytesseract.image_to_string(im_bw,lang="eng")
        temp=""
        for i in range(0,len(tempstr)):
            if re.match(reg, tempstr[i]):
                temp=temp+tempstr[i]
        outputs.append(temp)
        print("----String, Manual Threshold @ ",thresh,": ",outputs[-1])
        
        BestString=chooseBestString(outputs)
        
        CardNumber1=parse_1(BestString)
        print("Parsed Card Number (1): ",CardNumber1)
        
        CardNumber2=parse_2(BestString)
        print("Parsed Card Number (2): ",CardNumber2)



