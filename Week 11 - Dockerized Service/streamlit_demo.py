"""
To run this app, in your terminal:
> streamlit run streamlit_demo.py

Source: https://is.gd/SobJvL
"""

import streamlit as st
import pandas as pd
import joblib
from PIL import Image

# Loading model 
clf = joblib.load('./model/iris_classifier.joblib')

# Create title and sidebar
st.title("Iris flower species Classification App")
st.sidebar.title("Features")

# Loading images
setosa= Image.open('setosa.png')
versicolor= Image.open('versicolor.png')
virginica = Image.open('virginica.png')

# Intializing parameter values
parameter_list=['Sepal length (cm)','Sepal Width (cm)','Petal length (cm)','Petal Width (cm)']
parameter_input_values=[]
parameter_default_values=['5.2','3.2','4.2','1.2']
values=[]

# Display above values in the sidebar
for parameter, parameter_df in zip(parameter_list, parameter_default_values):
	values= st.sidebar.slider(label=parameter, key=parameter,value=float(parameter_df), min_value=0.0, max_value=8.0, step=0.1)
	parameter_input_values.append(values)
	
input_variables=pd.DataFrame([parameter_input_values],columns=parameter_list,dtype=float)
st.write('\n\n')


# Button that triggers the actual prediction
if st.button("Click Here to Classify"):
	prediction = clf.predict(input_variables)
	# Display the corresponding image based on the prediction made by the model
	# st.image(setosa) if prediction == 0 else st.image(versicolor)  if prediction == 1 else st.image(virginica)  # Breaks in 1.2.0
	if prediction == 0:
		st.image(setosa)
	elif prediction == 1:
		st.image(versicolor)
	else:
		st.image(virginica)