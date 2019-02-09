# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 02:08:32 2019

@author: Kunal
"""

import pytesseract
import imageio
import cv2
import numpy as np

#------------------------------------------------------------------------------
def getMaxIdx(values):
    return values.index(max(values))

def Scan():
    cam = cv2.VideoCapture(0)

    cv2.namedWindow("test")
    
    img_counter = 0
    
    while True:
        ret, frame = cam.read()
        cv2.imshow("test", frame)
        if not ret:
            break
        k = cv2.waitKey(1)
    
        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif k%256 == 32:
            # SPACE pressed
            img_name = "I_{}.png".format(img_counter)
            cv2.imwrite(img_name, frame)
            img=frame
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
    
#------------------------------------------------------------------------------
pytesseract.pytesseract.tesseract_cmd = r'C:\!Kunal\Tesseract-OCR\tesseract.exe'



#img=Scan()
img = cv2.imread('I_00.png',-1)
#img=cv2.resize(img,(800,500))

display(img)
img=Gamma_correction(img)
display(img)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

'''gaussian thresholding'''
(thresh, im_gauss) = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

#img = cv2.medianBlur(img, 3)
'''manual thresholding'''
#Bigger number more black, Smaller number more white. Start at 70 till you get a good image
thresh = 70
im_bw = cv2.threshold(img_gray, thresh, 255, cv2.THRESH_BINARY)[1]

display(im_bw)

outputs=[]
outputs.append(pytesseract.image_to_string(img,lang="eng"))
print("String 1, Image normal: ",outputs[-1])

outputs.append(pytesseract.image_to_string(img_gray,lang="eng"))
print("String 2, Image gray: ",outputs[-1])

outputs.append(pytesseract.image_to_string(im_gauss,lang="eng"))
print("String 3, Gaussian Threshold: ",outputs[-1])

outputs.append(pytesseract.image_to_string(im_bw,lang="eng"))
print("String 4, Manual Threshold: ",outputs[-1])
'''
card=r'[0-9][0-9][0-9][0-9]'

result=re.match(card,str4,re.I)
print(result.group())
'''
#import string

strlen=[]
for op in outputs:
    strlen.append(len(op))



idx=getMaxIdx(strlen)
Bstr=outputs[idx]

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
        
        
print("Parsed Card Number (Simple): ",card_no)

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

print("Method 2: ",card_num)
        


