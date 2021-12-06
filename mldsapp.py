# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 11:45:41 2021

@author: EricH
"""

import streamlit as st
from PIL import Image
import imageio

st.title('Eric Holland Deepfake Demo App')

st.header('25 possible combinations.')
st.markdown('Use the dropdown menus to generate your own deepfake using the source images and videos!')



## Resize image and video to 256x256
import dlib
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from imutils.face_utils import FaceAligner

pose_predictor=dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
fa = FaceAligner(pose_predictor)
face_encoder=dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')
detector = dlib.get_frontal_face_detector()
modelFile = 'opencv_face_detector_uint8.pb'
configFile = 'opencv_face_detector.pbtxt'
net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)



def load_image(image_file):
	img = Image.open(image_file)
	return img

y_n = ['Yes','No']
toggleupload = st.selectbox("Do you want to use your own file to see how GAN sees it?",y_n)

if toggleupload == "Yes":
    new_image = st.file_uploader("Upload Images", type=["png","jpg","jpeg"])
    if new_image is not None:
        new_image_show = Image.open(new_image)
        st.image(new_image_show)
    else: 
        st.write("Try uploading an image.")

else:
    new_image = False
    st.write("Try using one of the sample images below.")



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
