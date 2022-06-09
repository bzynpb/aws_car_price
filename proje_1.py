import streamlit as st
import numpy as np
import pandas as pd
import pickle

model = pickle.load(open("autoscout.pkl", "rb"))
enc = pickle.load(open("autoscout_encoder.pkl", "rb")) # 
df = pd.read_csv("final_model.csv")

st.markdown("# <center>AutoScout Car Price Predictor</center>", unsafe_allow_html=True )
col1, col2= st.columns(2) # ekrani iki kolona ayirdik

user_make_model = col1.selectbox("Select your car's Make&Model", df.make_model.unique())
# selectbox icinde yazacak yazi ve listede nereden neleri sececegini yazdik

user_body_type = col2.selectbox("Select your car's Body Type", df.body_type.unique())

user_gear = col1.selectbox("Select your car's Gearing Type", df["Gearing Type"].unique())
user_fuel = col2.selectbox("Select your car's Fuel Type", df.Fuel.unique())

user_km = col1.number_input("KM", 0, 300000, step=10000)
user_age = int(col2.selectbox("Age", (0,1,2,3)))

user_cc = col1.number_input("Displacement (cc)", 900, 2967, 1200, 100)
# degerleri esitleyerek de verebilirdik
# manuel olarak elle girdik 
user_hp = col2.number_input("HP", 55, 390, 90, 10)

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


c1, c2, c3, c4, c5,c6,c7,c8,c9 = st.columns(9) 
if c5.button('Predict'):
    result = model.predict(car)[0]
    st.info(f"Predicted value of your car : ${round(result)}")


