import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import pickle
from PIL import Image
import base64

# une petite documentation: https://fr.acervolima.com/un-guide-du-debutant-pour-streamlit/
#-------------image-Background----------------------------
def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    opacity: 0.8;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)

set_background('le-modèle-hexagonal.jpg')
#-------------image-sur la page-----------------------------------

image = Image.open('modeles-de-data-science.jpg')
st.image(image, caption='Data Science')

#--------------telecharger les "docs" de modeles-------------------

#st.title('Welcome to Diabetes Prediction Application') 
st.markdown('<p style="font-family:Times New Roman; text-align: center; font-style: oblique; font-size: 50px; font-weight: bold;color:MediumVioletRed ">Welcome to Diabetes Prediction Application</p>', unsafe_allow_html=True)


file1 = open('diabetes_prediction_knn.pkl', 'rb')
knn = pickle.load(file1)
file1.close()

file2 = open('diabetes_prediction_lg_reg.pkl', 'rb')
lg_reg = pickle.load(file2)
file2.close()

file3 = open('diabetes_prediction_dtc.pkl', 'rb')
dtc = pickle.load(file3)
file3.close()

file4 = open('diabetes_prediction_rfc.pkl', 'rb')
rfc = pickle.load(file4)
file4.close()


data = pd.read_csv("diabete_population.csv")
print(data)




#--------------------------------------------------------------------------
#pour le font et le size: mais ne fonctionne pas- 
#Enter_your_age = """
#    <style>
#        .stApp p{
#           font-family:Courier;
#          font-size: 20px;
#          color:Blue;
#          background: repeating-linear-gradient(-45deg, red 0%, yellow 7.14%, rgb(0,255,0) 14.28%, rgb(0,255,255) 21.4%, cyan 28.56%, blue #35.7%, magenta 42.84%, red 50%);
#          background-size: 600vw 600vw;
#       }
#   </style> 
    
#     """
#st.markdown(Enter_your_age, unsafe_allow_html=True) 
#-------------------------------------

#Enter_your_age = (<p style="font-family:Courier; color:Blue; font-size: 20px; font-weight: bold;">Enter your age </p>)
st.markdown('<p style="font-family:cursive; color:Blue; font-size: 20px; font-weight: bold;">Enter your age </p>', unsafe_allow_html=True)

# Pour normaliser chaque colonne, exécuter le code suivant: data_norm=(data[colonne]-moyenne)/ecart_type
age = st.number_input('')
age= (age- data['age'].mean())/data['age'].std()


#Nb_grossesses = '<p style="font-family:Courier; color:Blue; font-size: 20px; font-weight: bold;">Select the Number of grossesses </p>'
st.markdown( '<p style="font-family:cursive; color:Blue; font-size: 20px; font-weight: bold;">Select the Number of grossesses </p>', unsafe_allow_html=True)

#pour choisir le nb de grossessses sur un interval
grossesses = st.slider('', 0, 14) 
st.text('Selected: {}'.format(grossesses)) 
#grossesses = st.number_input("Enter your grossesses") 
grossesses=(grossesses- data['grossesses'].mean())/data['grossesses'].std()


#Enter_level_insuline = '<p style="font-family:Courier; color:Blue; font-size: 20px;">Enter your level insuline</p>'
st.markdown('<p style="font-family:cursive; color:Blue; font-size: 20px;font-weight: bold;">Enter your level insuline</p>', unsafe_allow_html=True)

insuline= st.number_input('Enter_level_insuline') 
insuline= (insuline- data['insuline'].mean())/data['insuline'].std()


#1. as sidebar menu

#with st.sidebar:
#    selected= option_menu(menu_title= "Main Menu",   #required
#    options= ["Home", "Projects", "Contact"],   #required
 #   icons= ["house","book","envelope"] ,  #optional
#    menu_icon= "cast",   #optional
#    default_index= 0,   #optional
#                             )
    
#2. horizontal menu 
selected= option_menu(menu_title= "Model Prediction Diabet",   #required
    options= ["KNeighborsClassifier","logistic_Regression", "randomForestClassifier", "DecisionTreeClassifier"],   #required
    icons= ["caret-right","caret-right","caret-right","caret-right"] ,  #optional
    menu_icon= "cast",   #optional
    default_index= 0,   #optional
    orientation= "horizontal",
    styles={
          "container": {"padding": "0!important", "background-color": "#fafafa"},
            "icon": {"color": "orange", "font-size": "25px"}, 
            "nav-link": {"font-size": "25px", "text-align": "left","font-style": "oblique", "margin":"0px", "--hover-color": "#eee"},
            "nav-link-selected": {"background-color": "LightSteelBlue"},
        })


#nouveau commentaire -------------------------------------------------------

if(st.button('Predict Diabete')): 
    if selected == "KNeighborsClassifier": 
        query = np.array([grossesses, age, insuline])
        query = query.reshape(1, 3)
        print(query)
        prediction = knn.predict(query)[0]
        st.title(f" Predicted value {prediction}  {selected}" )

    elif selected == "logistic_Regression":
        query = np.array([grossesses, age, insuline])
        query = query.reshape(1, 3)
        print(query)
        prediction = lg_reg.predict(query)[0]
        st.title(f" Predicted value {prediction}  {selected}" )

    elif selected == "DecisionTreeClassifier": 
        query = np.array([grossesses, age, insuline])
        query = query.reshape(1, 3)
        print(query)
        prediction = dtc.predict(query)[0]
        st.title(f" Predicted value {prediction}  {selected}" )
                 
    
    elif selected == "randomForestClassifier":
        query = np.array([grossesses, age, insuline])
        query = query.reshape(1, 3)
        print(query)
        prediction = rfc.predict(query)[0]
        st.title(f" Predicted value {prediction}  {selected}" )
   
    if prediction == 0:
        st.markdown('''
           <style>
            .root-container {
               background-color: green;
                opacity: 0.9;
            }
            .st-b7 {
                color: white;
                font-size=20px;
                font-weight: bold;
                text-align: center;

            }
          .css-nlntq9 {
             font-family: Times New Roman;
              
            }
            </style>
            ''', unsafe_allow_html=True)

        st.success("You have a good health!")        
    
       # st.success(st.markdown( '<p style="font-family:cursive; color:Blue; font-size: 20px; font-weight: bold;">Select the Number of grossesses </p>', unsafe_allow_html=True)) 
    else:
        st.warning("Be Carful, You're supposed to get diabet!") 
            
            
 #1) Améliorer votre page web en utilisant d'autres composants du module streamlit  (modifié)
# 2)Vous pouvez aussi créer d'autres modèles de machine learning avec logistic_Regression(), randomForestClassifier() et DecisionTreeClassifier() et les utiliser dans la page web
  

#DeltaGenerator(_root_container=0, _provided_cursor=LockedCursor(_root_container=0, _index=13, _parent_path=(), _props={'delta_type': 'markdown', 'last_index': None}), _parent=DeltaGenerator(_root_container=0, _provided_cursor=None, _parent=None, _block_type=None, _form_data=None), _block_type=None, _form_data=None)
    
