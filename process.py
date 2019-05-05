# -*- coding: utf-8 -*-
"""
Created on Sun May  5 22:27:37 2019

@author: KK
"""

from OCR.CardOCR import CardOCR
from FaceRecognition.FaceRecognition import FaceRecognition
import tkinter as tk
master = tk.Tk()
c=CardOCR(r'E:\!Kunal\Tesseract-OCR\tesseract.exe')
FR=FaceRecognition("models/28-04-2019evenlargermodel.h5","models/deploy.prototxt","models/res10_300x300_ssd_iter_140000.caffemodel")
identity="Kunal"
FR.RegisterId(identity)


#c.ReadImg("OCR/I_02.png")
c.Scan()
c.display()
while True:
    if(c.isBlur()==True):
        print("Image not captured/not captured properly. Press Spacebar when window is open to capture image.")
        c.Scan()
    else:
        break

number,expiry,valid=c.OCR()
print("Card Number: ",number," Expiry Date: ",expiry," Valid: ",valid) 

e1 = tk.Entry(master)
e1.insert('0', number)
e1.pack()

e1.focus_set()

e2=tk.Entry(master)
e2.insert('0',expiry)
e2.pack()

final=""

def callback():
    final=e1.get() # This is the text you may want to use later
    master.destroy()
    print(final)
    

b = tk.Button(master, text = "OK", width = 10, command = callback)
b.pack()

master.mainloop()

FR.capture()
while True:
    if(FR.isBlur()==True):
        print("Image not captured/not captured properly. Press Spacebar when window is open to capture image.")
        FR.capture()
    else:
        break

if(FR.VerifyId(identity)):
    print("person is ",identity)
else:
    print("person is not ",identity)