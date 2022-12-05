#Import all Libraries
import streamlit as  st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier


st.write(""" 
# Simple IRIS Flower Prediction Web App
This app predics the **Iris flower** type
""")


#Sidebar Section Code here:
st.sidebar.header('User Input Parameters')

def user_input_features():
    sepal_length = st.sidebar.slider('Sepal length', 4.3, 7.9, 5.4) #Sidebar slider name, min value, max value and default value
    sepal_width = st.sidebar.slider('Sepal width', 2.0, 4.4, 3.4) #Sidebar slider name, min value, max value and default value
    petal_length = st.sidebar.slider('Petal length', 1.0, 6.9, 1.3) #Sidebar slider name, min value, max value and default value
    petal_width = st.sidebar.slider('Petal width', 0.1, 2.5, 0.2) #Sidebar slider name, min value, max value and default value
    data = {'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width} #Placing all the selected values in a dictionary named as data
    features = pd.DataFrame(data, index=[0]) #Creating pandas dataframe using data dictionary and displaying the index 0 values only.
    return features

###

#Main Section Code here:
df = user_input_features() #Defining the pandas dataframe function using a variable named as df

st.subheader('User Input parameters')
st.write(df)

iris = datasets.load_iris() #Loading the iris dataset
X = iris.data #Difining the train data as variable X
Y = iris.target #Difining the test data as variable Y

clf = RandomForestClassifier() #Defining the random forest classifier using clf variable
clf.fit(X, Y) #Applying the classifier to create a training model using X and Y information.

prediction = clf.predict(df) #Make Prediction
prediction_proba = clf.predict_proba(df) #Provide Prediction Probability

st.subheader('Class labels and their corresponding index number')
st.write(iris.target_names) #Displaying the class labels and there respective index number

st.subheader('Prediction')
st.write(iris.target_names[prediction]) #Displaying the predicted class name
#st.write(prediction)

st.subheader('Prediction Probability') 
st.write(prediction_proba) #Displaying the Prediction Probability

###