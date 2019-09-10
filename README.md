# Bachelor in Engineering Final Year Project
Repository for my group's BE Project. Members: Kunal Kolhe, Pranav Raka, Aishwarya Karad,  Sakshi Agrawal. Project is to validate online transactions using face recognition.

## Models
download all required models from the link: https://drive.google.com/open?id=1sjc3BjXMZUFnOjA1d7047V9KLuon0cgY
copy all models in the BE-Project/models folder.
Required files:  
deploy.prototxt  
le.pickle  
liveness1.model  
res10_300x300_ssd_iter_140000.caffemodel  
28-04-2019evenlargermodel.h5  

## Synopsis:
Our objective was to create a prototype system which would simulate how online transactions would be processed in the future. Our vision was that using face recognition, Use input could be reduced to a great extent while validating Online Transactions. In India, after every transaction, you recieve an OTP on your phone which your have to enter online. We envisioned a system where the user holds the card up to the camera, the system reads the card and then looks at the camera and the system takes an image and verifies whether the person performing the transaction is the actual person who owns the card. 
Another aim of the project is to reduce the interaction required on part of the user. 

## Datasets and process used:
For the face recognition model, we used the AT&T dataset with 10 images each of 40 people's faces in different orientations. I made a face pair generator which made a dataset consisting of like pairs and unline pairs. There were about 1600 pairs from which 400 were pairs of same person and the remaining were randomly chosen dissimilar pairs. 
We fed this dataset to a siamese network. The loss function we used was the contrastive loss function. 
The training set accuracy was 99% and the test set accuracy was 95%. 
But the model was not robust under bad lighting conditions. To improve the robustness of the model, we tried using a gamma correction to preprocess the image before training. This gave us a marginal improvement of 1-2%.
