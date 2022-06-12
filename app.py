import streamlit as st
import requests
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras import Sequential
import numpy as np
import cv2
import os
import urllib
from streamlit_lottie import st_lottie
from streamlit_lottie import st_lottie_spinner
from PIL import Image

### ------------------------------------------------------------------------------------------------------------------------------------------
st.sidebar.header('User Input Features')
selected_image = st.sidebar.file_uploader('Upload image from PC',type=['png', 'jpg'],help='Type of image should be PNG or JPEG')

image_cover = Image.open('./logo-ps.png')
st.image(image_cover,use_column_width= True)

st.write("""
# Tooth decay identification Web App

***Tooth decay identification*** is a final thesis project of Khong Minh Duc from IU-VNU advised by Dr. Le Duy Tan.
This project will help to shorcut diganosis timing of dentists in decay-era

***Classes***: 
```python
- carries
- no-carries
```
""")

# Lottie
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

lottie_giveimage_sidebar = load_lottieurl('https://assets4.lottiefiles.com/packages/lf20_urbk83vw.json')
lottie_giveimage_url = "https://assets2.lottiefiles.com/packages/lf20_9p4kck7t.json"
lottie_giveimage = load_lottieurl(lottie_giveimage_url)

os.makedirs('model',exist_ok = True)

if not selected_image:
	with st.sidebar:
            	st_lottie(lottie_giveimage_sidebar, key = 'giveimage_sidebar',height=500)

	st_lottie(lottie_giveimage,key = 'giveme',height=400)
else:
	img = Image.open(selected_image)
	img = np.array(img.convert("RGB"))
	st.image(img)
	model_url = 'https://github.com/kmduc1012/Thesis-project-final_Khong-Minh-Duc/releases/download/best_model_Xception/best_model.1.h5'
	urllib.request.urlretrieve(model_url, os.path.join("model", "best_model.1.h5"))

	classes = ['carry','no-carry']
	#FIXME belong to best_model.h5
	base_model = tf.keras.applications.Xception(include_top=False, input_shape=(224, 224, 3), weights=None,classes=2)

	model = Sequential()
	model.add(base_model)
	model.add(Flatten())
	model.add(Dense(512, activation="relu"))
	model.add(Dropout(0.2))
	model.add(Dense(256, activation="relu"))
	model.add(Dropout(0.2))
	model.add(Dense(2, activation="softmax"))

	model.load_weights('model/best_model.h5')

	img = cv2.resize(img, (224, 224))
	result = model.predict(img.reshape(1, 224, 224, 3))
	max_prob = max(result[0])	
	class_ind = list(result[0]).index(max_prob)
	class_name = classes[class_ind]

	st.write(class_name)
	#Recomendation
	if class_name == 'carry':
		st.write('Recomendation: Using P/S S100 Pro with whitening P/S Toothpaste will help you improve their oral health.')
	else:
		st.write('Recomendation: Your oral health is good. However, you should to use P/S S100 Pro Electric toothbrush to maintain it.')









