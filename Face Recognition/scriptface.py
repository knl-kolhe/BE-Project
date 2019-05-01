# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 13:46:47 2019

@author: dell
"""

from FaceRecognition import FaceRecognition

FaceRecognition.help()

c=FaceRecognition(r'E:\Tesseract-OCR\tesseract.exe')
#c.Scan()

#c.ReadImg("credit_card_01.png")
#c.display()
face=c.capture()

while True:
    if(c.isBlur(face)==True):
        print("Image not captured/not captured properly. Press Spacebar when window is open to capture image.")
        c.capture()
    else:
        c.faceverify(face)
        break

#number,expiry,valid=c.OCR()
#print("Card Number: ",number," Expiry Date: ",expiry," Valid: ",valid) 
