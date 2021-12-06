# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 11:45:41 2021

@author: EricH
"""

import streamlit as st
from PIL import Image

import numpy as np
import os
import cv2
face_cascade = cv2.CascadeClassifier('face_detector.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')



def face_detection(inputimg):
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
	# Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
	# Draw rectangle around the faces
    for (x, y, w, h) in faces:
				 cv2.rectangle(img, (x, y), (x+w, y+h), (45, 45, 255), 3)
    return img,faces  #faces is for counting the number of detected faces

def smile_detection(inputimg):
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
	# Detect faces
    smiles = smile_cascade.detectMultiScale(gray, 1.1, 4)
    for (ex,ey,ew,eh) in smiles:
	        cv2.rectangle(img,(ex,ey),(ex+ew,ey+eh),(45, 45, 255),2)
    return img

def cartoon_detection(inputimg):
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
	# Detect faces
    gray = cv2.medianBlur(gray, 5)
    edge_places = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
    color_user = cv2.bilateralFilter(img, 9, 300, 300)
    finalcart = cv2.bitwise_and(color_user, color_user, mask=edge_places)
    return finalcart

def eyes_detection(inputimg):
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
	# Detect faces
    eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)
    for (ex,ey,ew,eh) in eyes:
	        cv2.rectangle(img,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    return img

def Blackwhiteoutline_detection(inputimg):
    img = cv2.GaussianBlur(inputimg, (11, 11), 0)
    cannoli = cv2.Canny(img, 100, 150)
    return cannoli




os.chdir("C:/Users/EricH/MachineLearning/MLSDFinal")
mode = ['Facial Recognition','Deepfake Generators']
toggleupload = st.sidebar.selectbox("What functions do you want to explore?",mode)

if toggleupload == "Facial Recognition":
    
    #List of possible functions to call
    features_to_detect = ["Original","Full Faces", "Eyes","Smile","Cartonize", "Black & White Outline"]
    #removed original from above list because I show the original as the column width before the transform listing
    
    #First sidebar selection that will pop asking which function to call
    feature_list = st.sidebar.selectbox("What facial feature do you want to detect?",features_to_detect)
    
    #overall streamlit title that shows on the main page instead of the sidebar
    st.title('Facial Recognition Demo App')
    st.header('Use your own photo!')
    st.markdown('Upload a photo below and a neural network will identify facial features!')
    
    #File uploader that gives us the new_image, used to create img and array that goes into functions
    new_image = st.file_uploader("Upload Images", type=["png","jpg","jpeg"])
    
    #url string if needed (delete later if functionality changes)
    urlsite = str(new_image)
    
    
    if new_image is not None:
        
        #import again because for some reason it doesnt work otherwise.
        import cv2
        #show the image to the user as the input and convert it for functions
        image = Image.open(new_image)
        #show image here
        st.image(image, caption='Input', use_column_width=True)
        
        #only way this worked was to use np array instead of Image package in Pillow
        img_array = np.array(image)
        
        #writing image to a jpg if needed (Maybe delete if functionality changes?)
        
        # img_written = cv2.imwrite('out.jpg', cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
        
        # this is the last call that gets pulled into every function below. Very important!
        img = cv2.cvtColor(img_array,1)
        
        #faces is the best and standard for this, if nothing works before Wednesday just use this one.
        if feature_list == "Full Faces":
            result_imagef,numface = face_detection(img)
            st.image(result_imagef, use_column_width=True) 
            
            #Use this as a counter for multiple face detection, couldve done just 1
            #ripped from cv2.face_detection readme on github by the guy who made the xml files
            #incredibly useful
            st.header("**The algorithm found {} face(s)**".format(len(numface)))
            
        #Eyes will draw blue rectangle around them, be sure to test before publish.
        elif feature_list == "Eyes":
            result_imageeye = eyes_detection(img)
            st.image(result_imageeye, use_column_width=True) 
        
        #smile is straightforward, same as eyes but uses different Scaler and XML file.
        elif feature_list == "Smile":
            result_imagesmile = smile_detection(img)
            st.image(result_imagesmile, use_column_width=True)
            
        #make sure to run this one before     
        elif feature_list == "Black & White Outline":
            result_imagebw = Blackwhiteoutline_detection(img)
            st.image(result_imagebw, use_column_width=True)
            
        #make sure to run this one before      
        elif feature_list == "Cartonize":
            result_imagecart = cartoon_detection(img)
            st.image(result_imagecart, use_column_width=True)
        #o
    #catches people that won't upload an image from getting a blank screen
    else: 
        st.markdown("Please upload an image of 1 or more people to begin.")

elif toggleupload == 'Deepfake Generators':
    st.title('Eric Holland Deepfake Demo App')
    st.header('25 possible combinations.')
    st.markdown('Use the dropdown menus to generate your own deepfake using the source images and videos!')

    #image dropdown selection for the user, generates the first half of output name
    image_label_full = "What do you want your input image to be?"
    image_options_full = ('Eric Holland', "Mona Lisa", 'Pete Davidson','Vladimir Putin',
                      'Margot Robbie')
    image_select = st.selectbox(image_label_full, image_options_full , index=0, key=None, help=None, on_change=None, kwargs=None)
    concatimg = str(image_select)
    
    # video dropdown selection for the user, generates second half of name output for final video.
    vid_label_full = "What target video do you want to use?"
    video_options_full = ('Eric Holland','Leonardo Dicaprio', "Barack Obama", 'Gal Gadot', "Donald Trump")
    video_select = st.selectbox(vid_label_full, video_options_full, index=0, key=None, help=None, on_change=None, kwargs=None)
    concatvideo = str(video_select)
    
    
    #Data dictionary for the drop down menus and their corresponding file locations for preview
    filelocator_img = {
        'Eric Holland' : 'C:/Users/EricH/MachineLearning/MLSDFinal/images/erhresized.png',
        "Mona Lisa" : 'images//monalisa.png',
        'Pete Davidson' : 'images/petedavidsonresized.png',
        'Vladimir Putin' : 'images/Putin.png',
        'Margot Robbie' : 'images/Margotresized.png'
    }
    
    #Data dictionary for the drop down menus and their corresponding video files for previewing.
    filelocator_video = {
        'Eric Holland' : 'videos//myvidresized.mp4',
        'Leonardo Dicaprio' : 'videos//Dicaprio.mp4',
        "Barack Obama" : 'videos//Obama.mp4',
        'Gal Gadot' : 'videos//GalGadot.mp4',
        'Donald Trump' : 'videos//DT.mp4',
    }
    
    
    #Header row for clarity
    st.header("Here are your selections:")
    
    # # show the image to the user and blow it up to be the same width for consistency.
    chosen_image = filelocator_img[image_select]
    image_to_show = Image.open(chosen_image)
    st.image(image_to_show, caption = 'This is the image you selected', use_column_width = True)
    
    #show the video that the user selected. Note that st.video doesnt have a caption function, so did it by hand
    chosen_video = filelocator_video[video_select]
    video_file = open(chosen_video, 'r')
    st.video(chosen_video)
    st.caption('This is the video you selected.')
    
    
    #get concatenated name so you can call the dictionary for the final output location
    concat_name = concatimg + ' ' + concatvideo
    
    # pickled and referenced the final files directly in a dictionary
    # this speeds up the streamlit load time considerably, and makes it easier for calling the inputs
    
    output_locator = {
        'Eric Holland Eric Holland':"outputs//ERHErrorMsg.mp4",
        'Eric Holland Leonardo Dicaprio':'outputs//ERHDicaprio.mp4',
        'Eric Holland Barack Obama':'outputs//ERHObama.mp4',
        'Eric Holland Donald Trump':'outputs//ERHTrump.mp4',
        'Eric Holland Gal Gadot':'outputs//ERHGadot.mp4',
        'Mona Lisa Eric Holland':'outputs//MonaLisaERH.mp4',
        'Mona Lisa Leonardo Dicaprio': 'outputs//MonaLisaDicaprio.mp4',
        'Mona Lisa Barack Obama':'outputs//MonaLisaObama.mp4',
        'Mona Lisa Gal Gadot':'outputs//MonaLisaGadot.mp4',
        'Mona Lisa Donald Trump':'outputs//MonaLisaTrump.mp4',
        'Pete Davidson Eric Holland':'outputs//DavidsonERH.mp4',
        'Pete Davidson Leonardo Dicaprio':'outputs//DavidsonDicaprio.mp4',
        'Pete Davidson Barack Obama':'outputs//DavidsonObama.mp4',
        'Pete Davidson Gal Gadot':'outputs//DavidsonGadot.mp4',
        'Pete Davidson Donald Trump':'outputs//DavidsonTrump.mp4',
        'Vladimir Putin Eric Holland':'outputs//PutinERH.mp4',
        'Vladimir Putin Leonardo Dicaprio':'outputs//PutinDicaprio.mp4',
        'Vladimir Putin Barack Obama':'outputs//PutinObama.mp4',
        'Vladimir Putin Gal Gadot':'outputs//PutinGadot.mp4',
        'Vladimir Putin Donald Trump':'outputs//PutinTrump.mp4',
        'Margot Robbie Eric Holland':'outputs//RobbieERH.mp4',
        'Margot Robbie Leonardo Dicaprio':'outputs//RobbieDicaprio.mp4',
        'Margot Robbie Barack Obama':'outputs//RobbieObama.mp4',
        'Margot Robbie Gal Gadot':'outputs//RobbieGadot.mp4',
        'Margot Robbie Donald Trump':'outputs//RobbieTrump.mp4'
    }
    
    
    # notes for clarity before final output is generated
    st.markdown("The video will be transposed onto the target image to create a new video.")
    st.header("Here is your final output!")
    
    # grabs the path of the final output video from the dictionary and displays it
    chosen_output = output_locator[concat_name]
    video_file = open(chosen_output, 'r')
    st.video(chosen_output)
    
    #final explanation and thanks for using to show the user.
    st.markdown("Note that any preprocessed videos perform better than the raw images and videos of myself.")
    st.markdown("This is because the other images are part of the training files used to tune the GAN neural network.")
    st.markdown("The more training images, the easier the model deals with new variables like different angles or glasses.")
    st.header("Thank you for trying this deepfake simulator!")


# originalImage = cv2.imread("test-image.jpg")
# savedImage = cv2.imwrite("saved-test-image.jpg",originalImage)
# If you simply want to copy an image file however, there is no need to load it into memory, you can simply copy the file, without using opencv:

# from shutil import copyfile

# originalImage = "test-image.jpg"
# copyfile(originalImage,"saved-test-image.jpg")
