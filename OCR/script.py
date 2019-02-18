# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 11:09:41 2019

@author: KK
"""

#Object Oriented script for the script CardOCR
#
from CardOCR import CardOCR
CardOCR().Help()
c=CardOCR(r'C:\!Kunal\Tesseract-OCR\tesseract.exe')
c.Scan()
c.display()
while True:
    if(c.isBlur()==True):
        print("Image not captured/not captured properly. Press Spacebar when window is open to capture image.")
        c.Scan()
    else:
        break

c.OCR()
