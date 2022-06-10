import streamlit as st
import numpy as np
import pandas as pd
import pickle
from PIL import Image

model = pickle.load(open("autoscout.pkl", "rb"))
enc = pickle.load(open("autoscout_encoder.pkl", "rb")) # 
df = pd.read_csv("final_model.csv")


html_temp = """
<div style="background-color:navy;padding:1.5px">
<h1 style="color:white;text-align:center;">AutoScout Car Price Predictor </h1>
</div><br>"""
st.markdown(html_temp,unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4) # ekrani iki kolona ayirdik


# selectbox icinde yazacak yazi ve listede nereden neleri sececegini yazdik

# __________________________
user_body_type = col1.selectbox("Select Body Type", df.body_type.unique())
user_make_model = col3.selectbox("Select Model", df.make_model.unique())


# user_gear = col1.selectbox("Select your car's Gearing Type", df["Gearing Type"].unique())
user_fuel = col3.selectbox("Select Fuel Type", df.Fuel.unique())

user_gear = col1.selectbox("Select Gearing Type", df["Gearing Type"].unique())


user_km = col1.number_input("KM", 0, 300000, step=10000)
# user_km = st.sidebar.slider("Select KM", min_value=0, max_value=300000, step=10000)

user_age = int(col3.selectbox("Age", (0,1,2,3)))

user_cc = col1.slider("Displacement (cc)", min_value=900, max_value=2967, value=1200, step=100)

# degerleri esitleyerek de verebilirdik
# manuel olarak elle girdik 
user_hp = col3.slider("HP", min_value=55,  max_value=390, value=110, step=10)    

car = pd.DataFrame({"make_model" : [user_make_model],
                    "body_type" : [user_body_type],
                    "km" : [user_km],
                    "hp" : [user_hp],
                    "Gearing Type" : [user_gear],
                    "Displacement_cc" : [user_cc],
                    "Fuel": [user_fuel],
                    "Age" : [user_age]})



# kullanicidan alinan veriler ile df olusturduk

cat = car.select_dtypes("object").columns
car[cat] = enc.transform(car[cat])

# modeli olustururken kullandigimiz encodera gore numerik olarak degistiriyor 
# yukarida kullandigimiz enc yi kullaniyoruz ki veriler ve kategoriler birbirini tutsun 
    

with col2:
    st.write("")
    img = Image.open("{}.png".format(user_body_type))
    st.image(img, width=100)

with col4:
    img = Image.open("{}.png".format(user_make_model.split(" ")[0]))
    st.image(img, width=100)


st.write("")
st.write("")
st.write("")
st.write("")
html_temp = """
<div style="background-color:navy;padding:1.5px">
<h1 style="color:white;text-align:center;">Please Predict Your Value </h1>
</div><br>"""
st.markdown(html_temp,unsafe_allow_html=True)
c1, c2, c3, c4, c5,c6,c7,c8,c9 = st.columns(9) 
with c5:
    st.write("")
    st.write("")
    st.write("")

if c5.button('Predict Now!'):
    result = model.predict(car)[0]
    st.info(f"Predicted value of your car : ${round(result)}")



