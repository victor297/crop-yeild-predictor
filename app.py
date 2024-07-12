# Import Important Library.

import joblib
import streamlit as st 
from PIL import Image
import pandas as pd

# Load Model & Scaler & Polynomial Features
model = joblib.load('model.pkl')
sc = joblib.load('sc.pkl')
pf = joblib.load('pf.pkl')

# Load datasets
df_final = pd.read_csv('test.csv')
df_main = pd.read_csv('main.csv')

# Create a mapping to replace Ghana with Nigeria for display purposes
df_main_display = df_main.copy()
df_main_display['area'] = df_main_display['area'].replace('Ghana', 'Nigeria')

# Load Image
image = Image.open('img.png')

# Streamlit Function For Building Button & app
def main():
    st.image(image, width=650,)   
    html_temp = '''
    <div style='background-color:green; padding:12px'>
    <h1 style='color:  #fff; text-align: center;'>Yield Crop Prediction</h1>
    <p style='color:  #fff; text-align: center;'>By DAUDA SANNI ABACHA 20/47CS/01107'</p>
    </div>
    <h2 style='color:  green; text-align: center;'>Please Enter Input</h2>
    '''
    st.markdown(html_temp, unsafe_allow_html=True)

    country_display = st.selectbox("Type or Select a Country from the Dropdown.", df_main_display['area'].unique())
    crop = st.selectbox("Type or Select a Crop from the Dropdown.", df_main_display['item'].unique())
    average_rainfall = st.number_input('Enter Average Rainfall (mm-per-year).', value=None)
    presticides = st.number_input('Enter Pesticides per Tonnes Use (tonnes of active ingredients).', value=None)
    avg_temp = st.number_input('Enter Average Temperature (degree celcius).', value=None)
    
    # Convert display country back to original country for processing
    country = country_display.replace('Nigeria', 'Ghana')
    input_data = [country, crop, average_rainfall, presticides, avg_temp]
    result = ''
    if st.button('Predict', ''):
        result = prediction(input_data)
    temp = '''
     <div style='background-color:navy; padding:8px'>
     <h1 style='color: gold; text-align: center;'>{}</h1>
     </div>
     '''.format(result)
    st.markdown(temp, unsafe_allow_html=True)

# Prediction Function to predict from model
def update_columns(df, true_columns):
    df[true_columns] = True
    other_columns = df.columns.difference(true_columns)
    df[other_columns] = False
    return df

def prediction(input_data):
    categorical_col = input_data[:2]
    input_df = pd.DataFrame({'average_rainfall': input_data[2], 'presticides_tonnes': input_data[3], 'avg_temp': input_data[4]}, index=[0])
    input_df1 = df_final.head(1).iloc[:, 3:]
    true_columns = [f'Country_{categorical_col[0]}', f'Item_{categorical_col[1]}']
    input_df2 = update_columns(input_df1, true_columns)
    final_df = pd.concat([input_df, input_df2], axis=1).values
    test_input = sc.transform(final_df)
    test_input1 = pf.transform(test_input)
    predict = model.predict(test_input1)
    result = (int(((predict[0]/100) * 2.47105) * 100) / 100)
    return (f"The Production of Crop Yields: {result} quintal/acre yield production. "
            f"That means 1 acre of land produces {result} quintal of yield crop. It's all dependent on different parameters like average rainfall, average temperature, soil and many more.")

if __name__ == '__main__':
    main()
