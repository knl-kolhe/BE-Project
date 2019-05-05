# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 13:46:47 2019

@author: dell
"""

from FaceRecognition import FaceRecognition

FaceRecognition.help()

FR=FaceRecognition(r"E:\!Kunal\ML\FaceRecognition\28-04-2019evenlargermodel.h5")

#FR.display()

identity="Kunal"

FR.RegisterId(identity)


FR.capture()
while True:
    if(FR.isBlur()==True):
        print("Image not captured/not captured properly. Press Spacebar when window is open to capture image.")
        FR.capture()
    else:
        break


import time

start=time.time()

if(FR.VerifyId(identity)):
    print("person is ",identity)
else:
    print("person is not ",identity)

end=time.time()
print(end-start)
#FR.verifyID("KK")