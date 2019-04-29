# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 12:11:42 2019

@author: KK
"""

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

def display(img,string="Image"):
    cv2.imshow(string,img)
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
    
    '''
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
    '''
    
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

class CropLayer(object):
	def __init__(self, params, blobs):
		# initialize our starting and ending (x, y)-coordinates of
		# the crop
		self.startX = 0
		self.startY = 0
		self.endX = 0
		self.endY = 0

	def getMemoryShapes(self, inputs):
		# the crop layer will receive two inputs -- we need to crop
		# the first input blob to match the shape of the second one,
		# keeping the batch size and number of channels
		(inputShape, targetShape) = (inputs[0], inputs[1])
		(batchSize, numChannels) = (inputShape[0], inputShape[1])
		(H, W) = (targetShape[2], targetShape[3])

		# compute the starting and ending crop coordinates
		self.startX = int((inputShape[3] - targetShape[3]) / 2)
		self.startY = int((inputShape[2] - targetShape[2]) / 2)
		self.endX = self.startX + W
		self.endY = self.startY + H

		# return the shape of the volume (we'll perform the actual
		# crop during the forward pass
		return [[batchSize, numChannels, H, W]]

	def forward(self, inputs):
		# use the derived (x, y)-coordinates to perform the crop
		return [inputs[0][:, :, self.startY:self.endY,
				self.startX:self.endX]]

#------------------------------------------------------------------------------
pytesseract.pytesseract.tesseract_cmd = r'E:\!Kunal\Tesseract-OCR\tesseract.exe'
import re

'''
img=Scan()
print('here')
val=variance_of_laplacian(img)
print("Blurry Variance:",val)
blurthresh=600
if val<blurthresh:
    print("Image is blurry, Please Try Again.")
    img=None
'''

#net = cv2.dnn.readNetFromCaffe("holistically-nested-edge-detection\hed_model\deploy.prototxt", "holistically-nested-edge-detection\hed_model\hed_pretrained_bsds.caffemodel")

# register our new layer with the model
#cv2.dnn_registerLayer("Crop", CropLayer)

img = cv2.imread("I_02.png",-1)
#img = cv2.imread("credit_card_01.png",-1)

#img=cv2.resize(img,(800,500))
if img is None:
    print("Image not captured/not captured properly. Press Spacebar when window is open to capture image.")
else:
    (H, W) = img.shape[:2]

    print("[INFO] performing Canny edge detection...")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(gray, 30, 150)
    '''
    #blurred1=cv2.GaussianBlur(img,(5,5),0)
    #(H, W) = blurred1.shape[:2]
    blob = cv2.dnn.blobFromImage(img, scalefactor=1.0, size=(W, H),
    	mean=(104.00698793, 116.66876762, 122.67891434),
    	swapRB=False, crop=False)
    
    # set the blob as the input to the network and perform a forward pass
    # to compute the edges
    print("[INFO] performing holistically-nested edge detection...")
    net.setInput(blob)
    hed = net.forward()
    hed = cv2.resize(hed[0, 0], (W, H))
    hed = (255 * hed).astype("uint8")
    kernel = np.ones((5,5),np.uint8)
    '''
    display(img,"normal")
    #display(canny)
    #display(hed)
    '''
    display(hed)
    dilation = cv2.dilate(hed,kernel,iterations = 1)
    display(dilation)
    dilation = cv2.dilate(hed,kernel,iterations = 1)
    display(dilation)
    dilation = cv2.dilate(hed,kernel,iterations = 1)
    display(dilation)
     '''
    
    #img=Gamma_correction(img)
    #display(img)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    reg=r"(\d)|(/)|(\n)|( )"
    outputs=[]
    '''manual thresholding'''
    
    #Bigger number more black, Smaller number more white. Start at 70 till you get a good image
    for thresh in range(70,105,15):
        im_bw = cv2.threshold(img_gray, thresh, 255, cv2.THRESH_BINARY)[1]
        display(im_bw)
        #cv2.imwrite("Card_Image_01_"+str(thresh)+".png",im_bw)
        tempstr=pytesseract.image_to_string(im_bw,lang="eng")
        temp=""
        for i in range(0,len(tempstr)):
            if re.match(reg, tempstr[i]):
                temp=temp+tempstr[i]
        outputs.append(temp)
        print("----String, Manual Threshold @ ",thresh,": ",outputs[-1])
    '''   
    BestString=chooseBestString(outputs)
        
    CardNumber1=parse_1(BestString)
    print("Parsed Card Number (1): ",CardNumber1)
        
    CardNumber2=parse_2(BestString)
    print("Parsed Card Number (2): ",CardNumber2)
    '''
    
    #-----------------------------------------------canny--------------------------------------
    tempstr=pytesseract.image_to_string(canny,lang="eng")
    display(canny,"Canny")
    cv2.imwrite("Canny.jpg",canny)
    temp=""
    for i in range(0,len(tempstr)):
        if re.match(reg, tempstr[i]):
            temp=temp+tempstr[i]
    outputs.append(temp)
    #print("----String, Manual Threshold @ ",thresh,": ",outputs[-1])
    
    BestString=chooseBestString(outputs)#outputs[-1]#chooseBestString(outputs)
    print(BestString)
    CardNumber1=parse_1(BestString)
    print("Parsed Card Number (1): ",CardNumber1)
    
    CardNumber2=parse_2(BestString)
    print("Parsed Card Number (2): ",CardNumber2)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    #----------------------------------------------------------HED---------------------------
    '''
    tempstr=pytesseract.image_to_string(hed,lang="eng")
    display(hed,"HED" )
    temp=""
    for i in range(0,len(tempstr)):
        if re.match(reg, tempstr[i]):
            temp=temp+tempstr[i]
    outputs.append(temp)
    #print("----String, Manual Threshold @ ",thresh,": ",outputs[-1])
    print(temp,"hekki")
    BestString=temp#outputs[-1]#  chooseBestString(outputs)
    
    CardNumber1=parse_1(BestString)
    print("Parsed Card Number hed (1): ",CardNumber1)
    
    CardNumber2=parse_2(BestString)
    print("Parsed Card Number (2): ",CardNumber2)
    '''
    '''
    dilation = cv2.dilate(hed,kernel,iterations = 1)
    tempstr=""
    display(dilation)
    tempstr=pytesseract.image_to_string(dilation,lang="eng")    
    temp=""
    for i in range(0,len(tempstr)):
        if re.match(reg, tempstr[i]):
            temp=temp+tempstr[i]
    outputs.append(temp)
    #print("----String, Manual Threshold @ ",thresh,": ",outputs[-1])
    print(temp,"hekki")
    BestString=temp#outputs[-1]#  chooseBestString(outputs)
    
    CardNumber1=parse_1(BestString)
    print("Parsed Card Number hed (1): ",CardNumber1)
    
    CardNumber2=parse_2(BestString)
    print("Parsed Card Number (2): ",CardNumber2)
    
    dilation = cv2.dilate(dilation,kernel,iterations = 1)
    tempstr=""
    display(dilation)
    tempstr=pytesseract.image_to_string(dilation,lang="eng")    
    temp=""
    for i in range(0,len(tempstr)):
        if re.match(reg, tempstr[i]):
            temp=temp+tempstr[i]
    outputs.append(temp)
    #print("----String, Manual Threshold @ ",thresh,": ",outputs[-1])
    print(temp,"hekki")
    BestString=temp#outputs[-1]#  chooseBestString(outputs)
    
    CardNumber1=parse_1(BestString)
    print("Parsed Card Number hed (1): ",CardNumber1)
    
    CardNumber2=parse_2(BestString)
    print("Parsed Card Number (2): ",CardNumber2)
    
    dilation = cv2.dilate(dilation,kernel,iterations = 1)    
    tempstr=""
    display(dilation)
    tempstr=pytesseract.image_to_string(dilation,lang="eng")    
    temp=""
    for i in range(0,len(tempstr)):
        if re.match(reg, tempstr[i]):
            temp=temp+tempstr[i]
    outputs.append(temp)
    #print("----String, Manual Threshold @ ",thresh,": ",outputs[-1])
    print(temp,"hekki")
    BestString=temp#outputs[-1]#  chooseBestString(outputs)
    
    CardNumber1=parse_1(BestString)
    print("Parsed Card Number hed (1): ",CardNumber1)
    
    CardNumber2=parse_2(BestString)
    print("Parsed Card Number (2): ",CardNumber2)
    
    dilation = cv2.dilate(dilation,kernel,iterations = 1)
    tempstr=""
    display(dilation)
    tempstr=pytesseract.image_to_string(dilation,lang="eng")    
    temp=""
    for i in range(0,len(tempstr)):
        if re.match(reg, tempstr[i]):
            temp=temp+tempstr[i]
    outputs.append(temp)
    #print("----String, Manual Threshold @ ",thresh,": ",outputs[-1])
    print(temp,"hekki")
    BestString=temp#outputs[-1]#  chooseBestString(outputs)
    
    CardNumber1=parse_1(BestString)
    print("Parsed Card Number hed (1): ",CardNumber1)
    
    CardNumber2=parse_2(BestString)
    print("Parsed Card Number (2): ",CardNumber2)
    
    '''
    

