import streamlit as st
import pandas as pd
import numpy as np
import sklearn
import pickle

st.title("Product Price Prediction")
value = st.radio("Which Product's Price You Want To Predict?",['Laptop','Mobile','Car'])

if value == 'Laptop':
    pipe = pickle.load(open('pipe.pkl', 'rb'))
    df = pickle.load(open('data.pkl', 'rb'))

    st.header('Laptop Price Predictor')

    # Company
    Company = st.selectbox('Company',df['Company'].unique())
    # TypeName
    Type = st.selectbox('Type',df['TypeName'].unique())
    # Ram
    Ram = st.selectbox('Ram(GB)',df['Ram'].unique())
    # Weight
    Weight = st.number_input('Weight(kg)')
    # TouchScreen
    Touchscreen = st.selectbox('Touch Screen',['No','Yes'])
    # IPS
    IPS = st.selectbox('IPS',['No', 'Yes'])
    # ScreenSize
    Screen_size = st.number_input('Screen Size') 
    # Resolution
    Resolution = st.selectbox('Screen Resolution', ['1920x1080','1366x768','1600x900','3840x2160','3200x1800','2880x1800','2560x1600','2560x1440','2304x1440'])
    # cpu
    CPU = st.selectbox('CPU',df['CPU Brand'].unique())
    # hdd
    HDD = st.selectbox('HDD(in GB)',[0,128,256,512,1024,2048])
    # sdd
    SSD = st.selectbox('SSD(in GB)',[0,8,128,256,512,1024])
    # gpu
    GPU = st.selectbox('GPU',df['Gpu Brand'].unique())
    # os
    OS = st.selectbox('OS',df['OS name'].unique())


    if st.button('Predict Price'):
        # query
        ppi = None
        if Touchscreen == 'Yes':
            Touchscreen = 1
        else:
            Touchscreen = 0

        if IPS == 'Yes':
            IPS = 1
        else:
            IPS = 0

        X_res = int(Resolution.split('x')[0])
        Y_res = int(Resolution.split('x')[1])
        ppi = ((X_res**2) + (Y_res**2))**0.5/Screen_size
        query = np.array([Company,Type,Ram,Weight,Touchscreen,IPS,ppi,CPU,HDD,SSD,GPU,OS])

        query = query.reshape(1,12)
        st.title("The predicted price of this configuration is " + str(int(np.exp(pipe.predict(query)[0]))))

###################################################################################################################
        
elif value == 'Mobile':
    pipe = pickle.load(open('pipe_mobile.pkl', 'rb'))
    df = pickle.load(open('data_mobile.pkl', 'rb'))

    st.header('Mobile Price Predictor')

    # Company
    Company = st.selectbox('Company',sorted(df['Brand'].unique()))
    # MobileSize
    MobileSize = st.select_slider('Mobile Size',options = sorted(df['Mobile_Size'].unique()))
    # Ratings
    Ratings = st.select_slider('Ratings',options = sorted(df['Ratings'].unique()))
    # Ram
    Ram = st.select_slider('RAM',options = sorted(df['RAM'].unique()))
    # ROM
    Rom = st.select_slider('ROM',options = sorted(df['ROM'].unique()))
    # PrimaryCamera
    PrimaryCamera = st.select_slider('Primary Camera',options = sorted(df['Primary_Cam'].unique()))
    # Selfi_Cam
    Selfi_Cam = st.select_slider('Selfi Camera', options = sorted(df['Selfi_Cam'].unique()))
    # Battery
    Battery = st.select_slider('Battery',options = sorted(df['Battery_Power'].unique()))


    if st.button('Predict Price'):
        query = np.array([Ratings, Ram, Rom, MobileSize, PrimaryCamera, Selfi_Cam, Battery, Company])

        query = query.reshape(1,8)
        st.title("The predicted price of this configuration is " + str(int(np.exp(pipe.predict(query)[0]))))


###################################################################################################################

elif value == 'Car':
    pipe = pickle.load(open('ExtraTreesRegressorModel.pkl', 'rb'))
    df = pickle.load(open('cardata.pkl', 'rb'))

    st.header('Car Price Predictor')


    # Model Name
    name = st.selectbox('Model Name',sorted(df['name'].unique())) 
    # Company
    company = st.selectbox('Company',sorted(df['company'].unique()))
    # Year
    year = st.selectbox('Year',sorted(df['year'].unique()))
    # KM Drived
    km = st.number_input("KM's Driven")
    # Fuel Type
    fuel = st.selectbox('Fuel Type',['Petrol', 'Diesel', 'LPG'])

    if st.button('Predict Price'):
        query = np.array([name, company, year, km, fuel])

        query = query.reshape(1,5)
        st.title("The predicted price of this configuration is " +
                 str(int(pipe.predict(pd.DataFrame(columns=['name','company','year','kms_driven','fuel_type'],data=query))[0])))



