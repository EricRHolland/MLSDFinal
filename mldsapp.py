# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 11:45:41 2021

@author: EricH
"""

import streamlit as st
from PIL import Image
import numpy as np
st.title('Eric Holland Deepfake Demo App')
st.header('This app gives you 25 possible outputs of different types of deepfakes')
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
    "Mona Lisa" : 'C:/Users/EricH/MachineLearning/MLSDFinal/images/monalisa.png',
    'Pete Davidson' : 'C:/Users/EricH/MachineLearning/MLSDFinal/images/petedavidsonresized.png',
    'Vladimir Putin' : 'C:/Users/EricH/MachineLearning/MLSDFinal/images/Putin.png',
    'Margot Robbie' : 'C:/Users/EricH/MachineLearning/MLSDFinal/images/Margotresized.png'
}

#Data dictionary for the drop down menus and their corresponding video files for previewing.
filelocator_video = {
    'Eric Holland' : 'C:/Users/EricH/MachineLearning/MLSDFinal/videos/myvidresized.mp4',
    'Leonardo Dicaprio' : 'C:/Users/EricH/MachineLearning/MLSDFinal/videos/Dicaprio.mp4',
    "Barack Obama" : 'C:/Users/EricH/MachineLearning/MLSDFinal/videos/Obama.mp4',
    'Gal Gadot' : 'C:/Users/EricH/MachineLearning/MLSDFinal/videos/GalGadot.mp4',
    'Donald Trump' : 'C:/Users/EricH/MachineLearning/MLSDFinal/videos/DT.mp4',
}   


#Header row for clarity
st.header("Here are your selections:")


# show the image to the user and blow it up to be the same width for consistency.
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
    'Eric Holland Eric Holland':"C:/Users/EricH/MachineLearning/MLSDFinal/outputs/ERHErrorMsg.mp4",
    'Eric Holland Leonardo Dicaprio':'C:/Users/EricH/MachineLearning/MLSDFinal/outputs/ERHDicaprio.mp4',
    'Eric Holland Barack Obama':'C:/Users/EricH/MachineLearning/MLSDFinal/outputs/ERHObama.mp4',
    'Eric Holland Donald Trump':'C:/Users/EricH/MachineLearning/MLSDFinal/outputs/ERHTrump.mp4',
    'Eric Holland Gal Gadot':'C:/Users/EricH/MachineLearning/MLSDFinal/outputs/ERHGadot.mp4',
    'Mona Lisa Eric Holland':'C:/Users/EricH/MachineLearning/MLSDFinal/outputs/MonaLisaERH.mp4',
    'Mona Lisa Leonardo Dicaprio': 'C:/Users/EricH/MachineLearning/MLSDFinal/outputs/MonaLisaDicaprio.mp4',
    'Mona Lisa Barack Obama':'C:/Users/EricH/MachineLearning/MLSDFinal/outputs/MonaLisaObama.mp4',
    'Mona Lisa Gal Gadot':'C:/Users/EricH/MachineLearning/MLSDFinal/outputs/MonaLisaGadot.mp4',
    'Mona Lisa Donald Trump':'C:/Users/EricH/MachineLearning/MLSDFinal/outputs/MonaLisaTrump.mp4',
    'Pete Davidson Eric Holland':'C:/Users/EricH/MachineLearning/MLSDFinal/outputs/DavidsonERH.mp4',
    'Pete Davidson Leonardo Dicaprio':'C:/Users/EricH/MachineLearning/MLSDFinal/outputs/DavidsonDicaprio.mp4',
    'Pete Davidson Barack Obama':'C:/Users/EricH/MachineLearning/MLSDFinal/outputs/DavidsonObama.mp4',
    'Pete Davidson Gal Gadot':'C:/Users/EricH/MachineLearning/MLSDFinal/outputs/DavidsonGadot.mp4',
    'Pete Davidson Donald Trump':'C:/Users/EricH/MachineLearning/MLSDFinal/outputs/DavidsonTrump.mp4',
    'Vladimir Putin Eric Holland':'C:/Users/EricH/MachineLearning/MLSDFinal/outputs/PutinERH.mp4',
    'Vladimir Putin Leonardo Dicaprio':'C:/Users/EricH/MachineLearning/MLSDFinal/outputs/PutinDicaprio.mp4',
    'Vladimir Putin Barack Obama':'C:/Users/EricH/MachineLearning/MLSDFinal/outputs/PutinObama.mp4',
    'Vladimir Putin Gal Gadot':'C:/Users/EricH/MachineLearning/MLSDFinal/outputs/PutinGadot.mp4',
    'Vladimir Putin Donald Trump':'C:/Users/EricH/MachineLearning/MLSDFinal/outputs/PutinTrump.mp4',
    'Margot Robbie Eric Holland':'C:/Users/EricH/MachineLearning/MLSDFinal/outputs/RobbieERH.mp4',
    'Margot Robbie Leonardo Dicaprio':'C:/Users/EricH/MachineLearning/MLSDFinal/outputs/RobbieDicaprio.mp4',
    'Margot Robbie Barack Obama':'C:/Users/EricH/MachineLearning/MLSDFinal/outputs/RobbieObama.mp4',
    'Margot Robbie Gal Gadot':'C:/Users/EricH/MachineLearning/MLSDFinal/outputs/RobbieGadot.mp4',
    'Margot Robbie Donald Trump':'C:/Users/EricH/MachineLearning/MLSDFinal/outputs/RobbieTrump.mp4'
}


# notes for clarity before final output is generated
st.markdown("The video will be transposed onto the target image to create a new video.")
st.header("Here is your final output!")

#grabs the path of the final output video from the dictionary and displays it
chosen_output = output_locator[concat_name]
video_file = open(chosen_output, 'r')
st.video(chosen_output)

#final explanation and thanks for using to show the user. 
st.markdown("Note that any preprocessed videos perform better than the raw images and videos of myself.")
st.markdown("This is because the other images are part of the training files used to tune the GAN neural network.")
st.markdown("The more training images, the easier the model deals with new variables like different angles or glasses.")
st.header("Thank you for trying this deepfake simulator!")



