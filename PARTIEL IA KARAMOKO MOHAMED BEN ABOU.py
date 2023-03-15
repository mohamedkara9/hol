from nbconvert.writers.base import WriterBase


class HelloWriter(WriterBase):
    def write(self, output, resources, notebook_name=None, **kw):
        with open("hello.txt", "w") as outfile:
            outfile.write("hello world")
import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


st.title('Projet IA Karamoko Mohamed Ben Abou')
st.subheader('Master 1 Informatique 2022-2023')
st.markdown('JEU DE DONNEE:')
def load_data():
   bikes_data_path = Path() / 'C:/Users/moham/Downloads/heart.csv'
   data = pd.read_csv(bikes_data_path)
   return data

df = load_data()
st.write(df)

if st.button('VOIR LE NOMBRE DE LIGNE ET COLONNE DU JEU:'):
   
    st.write("nombre de ligne : ", df.shape[0])
    st.write("numbre de colonne : ", df.shape[1])


if st.button('VOIR LES DONNEES MANQUANTES DU TABLEAU:'):
   
    st.markdown('Nombre total de valeurs manquantes:')
    # Nombre total de valeurs manquantes
    st.write(df.isnull().sum().sum())

if st.button('VOIR LE NOM DES COLONNES:'):
    df.columns
if st.button('STATISTIQUES DESCRIPTIVES:'):
#Afficher un peu de statistiques descriptives de toutes les variables numériques
   st.write(df.describe())
if st.button('VOIR LE TYPE DES COLONNES:'):
# Check column types
   st.write(df.dtypes)
    


st.markdown('À propos de l''ensemble de données : âge : Âge du patient sexe : Sexe du patient cp : type de douleur thoracique, 0 = angine typique, 1 = angine atypique, 2 = douleur non angineuse, 3 = asymptomatique trtbps : tension artérielle au repos (en mm Hg) chol : Cholestoral en mg/dl récupéré via le capteur IMC fbs : (glycémie à jeun > 120 mg/dl), 1 = Vrai, 0 = Faux restecg : résultats électrocardiographiques au repos, 0 = normal, 1 = normalité de l''onde ST-T, 2 = hypertrophie ventriculaire gauche thalachh : Fréquence cardiaque maximale atteinte oldpeak : pic précédent slp : pente caa : Nombre de navires principauxthall : résultat du test de stress au thalium ~ (0,3)exng : angine de poitrine induite par l''effort ~ 1 = oui, 0 = non sortie : Variable cible 0 : "BAS", 1 :"ÉLEVÉ"')



import pickle
import os
import numpy as np
import streamlit as st

MMS_SAVE_PATH = os.path.join(os.getcwd(),'saved_path', 'mms_scaler.pkl')
MODEL_PATH = os.path.join(os.getcwd(), "saved_path", "model.pkl")



heartattack_chance = {0:"faible", 1:"forte"}




with st.form('Heart Attack Prediction Form'):
    st.title("Prédiction de crise cardiaque")
    st.header("Informations sur le patient")
    age = int(st.number_input("Age:")) # add int because not float
    sex = int(st.number_input("Sex") )
    cp = int(st.number_input("Type de douleur thoracique :"))
    st.caption('''
        Valeur 1:  angine typique \n
        Valeur 2: angine atypique \n
        Valeur 3: douleur non angineuse \n
        Valeur 4: asymptomatique \n
             ''')
    trtbps = st.number_input("Tension artérielle au repos (en mm Hg):") # not need int because of float value
    chol = st.number_input("Cholesterol (in mg/dl):")
    fbs = int(st.number_input("Glycémie à jeun:"))
    st.caption('''
        Valeur 1 => 120 mg/dl \n
        Valeur 0 < 120 mg/dl \n
             ''')
    restecg = int(st.number_input("Électrocardiographique de repos :"))
    st.caption('''
        Valeur 0: normale\n
        Valeur 1: ayant une anomalie de l'onde ST-T \n
        Valeur 2: montrant une hypertrophie ventriculaire gauche probable ou certaine \n
             ''')
    thalachh = st.number_input("Fréquence cardiaque maximale :")
    exng = int(st.number_input("Exercice pendant angine:"))
    st.caption('''
        Valeur 1 = Oui \n
        Valeur 0 = Non \n
             ''')
    oldpeak = st.number_input("Pic précédent:")
    slp = int(st.number_input("Pente:"))
    caa = int(st.number_input("Nombre de navires principaux (0-3) :"))
    thall = int(st.number_input("Taux Thal :"))
    
    submitted = st.form_submit_button('✅ envoyer')
    
    if submitted == True:
        patient_info = np.array([age,sex,cp,trtbps,chol,fbs,restecg,thalachh,exng,oldpeak,slp,caa,thall])
        patient_info = mms_scaler.transform(np.expand_dims(patient_info, axis=0))
        new_pred = classifier.predict(patient_info)
        if np.argmax(new_pred) == 1:
            st.warning
            (f'''Avertissement! Ce patient a  {heartattack_chance[np.argmax(new_pred)]} 
              risques de crise cardiaque ☹️ ''')
        else:
            st.snow()
            st.success
            (f'''Ce patient a  {heartattack_chance[np.argmax(new_pred)]} 
              risques de crise cardiaque 🙂''')
             
