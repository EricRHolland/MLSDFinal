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



# using base directory to see if it runs locally before deploying.

# os.chdir("C:/Users/EricH/MachineLearning/MLSDFinal")
mode = ['Facial Recognition','Deepfake Generators','App Summary']
toggleupload = st.sidebar.selectbox("What functions do you want to explore?",mode)

if toggleupload == "Facial Recognition":
    
    #List of possible functions to call
    features_to_detect = ["Select an option","Full Faces", "Eyes","Smile","Cartonize", "Black & White Outline"]
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
    # urlsite = str(new_image)
    
    
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
    st.title('Deepfake Demo App')
    st.markdown('Use the dropdown menus to generate your own deepfake using the source images and videos!')

    #image dropdown selection for the user, generates the first half of output name
    image_label_full = "What do you want your input image to be?"
    image_options_full = ('Eric Holland', "Mona Lisa", 'Pete Davidson','Vladimir Putin',
                      'Margot Robbie')
    image_select = st.selectbox(image_label_full, image_options_full , index=0, key=None, help=None, on_change=None, kwargs=None)
    concatimg = str(image_select)
    
    # video dropdown selection for the user, generates second half of name output for final video.
    vid_label_full = "What target video do you want to use?"
    video_options_full = ('Eric Holland','Leonardo Dicaprio', "Barack Obama", 'Elodie Yung', "Donald Trump")
    video_select = st.selectbox(vid_label_full, video_options_full, index=0, key=None, help=None, on_change=None, kwargs=None)
    concatvideo = str(video_select)
    
    
    #Data dictionary for the drop down menus and their corresponding file locations for preview
    filelocator_img = {
        'Eric Holland' : 'images//erhresized.png',
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
        'Elodie Yung' : 'videos//GalGadot.mp4',
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
        'Eric Holland Elodie Yung':'outputs//ERHGadot.mp4',
        'Mona Lisa Eric Holland':'outputs//MonaLisaERH.mp4',
        'Mona Lisa Leonardo Dicaprio': 'outputs//MonaLisaDicaprio.mp4',
        'Mona Lisa Barack Obama':'outputs//MonaLisaObama.mp4',
        'Mona Lisa Elodie Yung':'outputs//MonaLisaGadot.mp4',
        'Mona Lisa Donald Trump':'outputs//MonaLisaTrump.mp4',
        'Pete Davidson Eric Holland':'outputs//DavidsonERH.mp4',
        'Pete Davidson Leonardo Dicaprio':'outputs//DavidsonDicaprio.mp4',
        'Pete Davidson Barack Obama':'outputs//DavidsonObama.mp4',
        'Pete Davidson Elodie Yung':'outputs//DavidsonGadot.mp4',
        'Pete Davidson Donald Trump':'outputs//DavidsonTrump.mp4',
        'Vladimir Putin Eric Holland':'outputs//PutinERH.mp4',
        'Vladimir Putin Leonardo Dicaprio':'outputs//PutinDicaprio.mp4',
        'Vladimir Putin Barack Obama':'outputs//PutinObama.mp4',
        'Vladimir Putin Elodie Yung':'outputs//PutinGadot.mp4',
        'Vladimir Putin Donald Trump':'outputs//PutinTrump.mp4',
        'Margot Robbie Eric Holland':'outputs//RobbieERH.mp4',
        'Margot Robbie Leonardo Dicaprio':'outputs//RobbieDicaprio.mp4',
        'Margot Robbie Barack Obama':'outputs//RobbieObama.mp4',
        'Margot Robbie Elodie Yung':'outputs//RobbieGadot.mp4',
        'Margot Robbie Donald Trump':'outputs//RobbieTrump.mp4'
    }
    
    
    # notes for clarity before final output is generated
    st.markdown("The video will be transposed onto the target image to create a new video.")
    st.header("Deepfake Output:")
    
    # grabs the path of the final output video from the dictionary and displays it
    chosen_output = output_locator[concat_name]
    video_file = open(chosen_output, 'r')
    st.video(chosen_output)
    
    #final explanation and thanks for using to show the user.
    st.markdown("Note that any preprocessed videos perform better than the raw images and videos of myself.")
    st.markdown("This is because the other images are part of the training files used to tune the GAN neural network.")
    st.markdown("The more training images, the easier the model deals with new variables like different angles or glasses.")
    st.subheader("Thank you for trying this deepfake simulator!")


# st.write(row_dict['hotelName'].values[0] , "\n")

elif toggleupload == 'App Summary':
    st.title("Original Vision")
    st.write("This app was originally designed to be one where you upload a file and a GAN neural network will create a deepfake for you.",
             "However, as the project materialized, it became clear that Streamlit functionality with large neural network models was cumbersome and unreliable.")
    st.write("As a result, the project was split into two parts: a sample of deepfake generators and a separate facial recognition app that showcases GAN.",
             "These two app ideas were formed into 1 via a sidebar that toggles between the two. I've provided a short writeup of how I made this below.")
    
    st.header("Deepfake Generator App Process")
    st.write("The original process was running a Google Colab with a pretrained model and allowed users to combine their own image with a sample video.",
             "Some of the packages including ffmpeg have no streamlit functionality, so after a lot of failed attempts to still use the same app workflow, I ended up opting for a sample dictionary model.",
             "This deepfake relies on calling a dictionary of pre-fabricated combinations that are stored in my Github folder for this project. It helped me learn more about pickling and how to format classes and dictionaries for clean code.",
             "The image and video dictionaries build on one another, with the concatenation of both the .img and .vid creating the search code for the combination output file. This helped a lot with organizing my results.",
             "Changing my approach was challenging but I learned how to deal with exceptions and new python packages as a result.")    

    #how are the deepfakes generated?
    st.subheader("Deepfake Generator App Retrospective")
    st.write("The training images are diversified between those within the model training set (Mona Lisa, Putin) and images generated by me or cropped after a Google Search.",
             "Margot Robbie and Pete Davidson are from internet searches, and I uploaded my own photo. The performance of the models on the images in the trained set is noticeably better.",
             "In particular, the encoding of my video to a 256x256 size to comply with the system constraints squished and muddied the image.")
    st.write("The deepfake videos here do poorly when they are asked to move the eyes away from the camera and also have relative movement. Notice how all the source images perform worse with my video.",
             "This is because the best performing videos are keeping their eyes or point of focus constant, and their encoding of different facial features is fine-tuned."
             "After trying for days to get the GAN to work on Streamlit, I decided to move to an output call model where all the images and videos are stored in Github and callable by dictionary.",
             "This greatly improved runtime but removed the potential for a user to create their own deepfake. This feature will come in a future version.")


    st.header("Facial Recognition App Process")
    st.write("The process of the facial feature recognition process is very basic. I'm calling a series of XML files that have the pretrained models in them. The uploaded imaged is parsed by pillow Image package and transformed into a numpy array.",
             "After being converted to a numpy array, it is read as an image through the openCV package, a very popular and powerful image processing package.")
    st.write("Once the openCV package reads the image, it calls a given XML model file based on what the user selects and applies it to the image they upload."
             "Once it calls the model, the output is again relatively simple. It reprocesses the transformed image with a PIL write function since CV2.imwrite() wouldn't work correctly in Streamlit.")
    
        
    st.subheader("Facial Recognition App Retrospective")
    st.write("Facial feature recognition is a precusor to deepfake technology, and I wanted the user to see how a relatively old model performs on newer images or file types that didnt exist 8 years ago.",
             "The facial recognition model is able to identify different facial features like eyes and smiles as well as full faces.",
             "Once, again, these pre trained models require some image processing to do very well (downsizing image files to 256x256 and preferably a uniform or blurred background).")
    st.write("The output works well with full faces that are facing toward the camera completely. The eyes still do well, but the final 3 XML models are less accurate. This is because their training data was much smaller than the full face recognition dataset.",
             "The models were harder than anticipated to implement in Streamlit but I learned how to implement image processing and rewrites without any prior knowledge of image use in Python or Streamlit.",
             "Since I didn't have any prior knowledge for image and video handling, this project was very challenging to learn and implement with independent study.")
    
    tensorflow_old_demo = 'images//GAN-demo.gif'
    st.header("Problems Encountered & Future Additions")
    st.write("One issue I was having is that many of the results for similar GAN and open-CV papers created similar apps on Google Colab or React (a platform similar to streamlit).",
             "I originally envisioned an app similar to this short demo shown below:")
    st.image(tensorflow_old_demo)
    st.markdown("So why couldn't I just create a new implementation of the demo image shown above?")
    st.write("The main reason is that the majority of GAN and facial detection/recognition software and code is not meant for active app deployment, as I found out.",
             "This specific demo runs on a deprecated version of tensorflow and an old version of Python.",
             "After spending 2 days trying to get CV2 and other packages to work on what I had so far, I didn't want to risk installing a new version of python before the deadline.",
             "It was clear that trying to learn Streamlit implementation for tensorflow with no prior knowledge 2 days before a deadline was not going to work, so I had to adapt.",
             "Overall, I'm happy with how much I pushed myself to learn things that I was completely clueless about just 6 days ago.")
    st.subheader("Thank you for your time.")

        
# originalImage = cv2.imread("test-image.jpg")
# savedImage = cv2.imwrite("saved-test-image.jpg",originalImage)
# If you simply want to copy an image file however, there is no need to load it into memory, you can simply copy the file, without using opencv:

# from shutil import copyfile

# originalImage = "test-image.jpg"
# copyfile(originalImage,"saved-test-image.jpg")
