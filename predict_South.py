import streamlit as st
from joblib import load
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
from sklearn.preprocessing import OneHotEncoder


# Load the pipeline model
@st.cache_data
def load_model():
    pipeline = load('best_model_SA.joblib')
    return pipeline

pipeline_NA = load_model()

# Define the Streamlit page
def show_predict_page_south():
    st.markdown("<h1 style='text-align: center; color: red;'>Salary Prediction in South America for Software Engineers</h1>", unsafe_allow_html=True)
    st.write("""We need some information to predict the salary""")

    countries = (
        'Brazil',
        'Argentina',
        'Colombia',
        'Chile',
        'Uruguay',
        'Peru',
        'Ecuador',
        'Paraguay',
        'Bolivia',
        'Guyana',
        'Suriname',
    )

    education = (
        "Less than a Bachelors",
        "Bachelor’s degree",
        "Master’s degree",
        "Post grad",
    )

    # User inputs
    country = st.selectbox("Country", countries)
    education_level = st.selectbox("Education Level", education)
    experience = st.slider("Years of Experience", 0, 50, 3)

    # Calculate Salary button
    ok = st.button("Calculate Salary")
    if ok:
        # Create a dataframe for the inputs
        X_new = pd.DataFrame({
            'Country': [country],
            'Education Level': [education_level],
            'Years of Experience': [experience]
        })

        # Make prediction
        salary = pipeline_NA.predict(X_new)
        st.subheader(f"The estimated salary is \\${salary[0]:.2f}")



      
##############FIN PREDICION SALARIO######################
 
    def load_data():
        df = pd.read_csv("south_america.csv")
        df = df[["Country", "Education Level", "Years of Experience", "Salary"]]
        return df
    

    df = load_data()

    st.write("""""")
    st.write("""""")
    st.write("""""")

    st.markdown("<h2 style='text-align: center;';>Continue exploring about the salary landscape in South America for Software Engineers</h2>", unsafe_allow_html=True)


    st.write("""""")
    st.write("""""")
    st.write("""""")
    st.write("""""")
    st.write("""""")
    
    
    st.write("""#### % of the Data surveyed from different countries in South America""")
    # st.pyplot(fig1)
    data = df["Country"].value_counts()
    fig = px.pie(data, values=data.values, names=data.index)
    fig.update_traces(textposition='inside', textinfo='percent+label')
    # Set the height of the chart, this will make the pie chart bigger. You can adjust the height value as needed.
    fig.update_layout(height=800)
    st.plotly_chart(fig, use_container_width=True)
    
    
    st.write("#### Mean Salary Based On Country in South America")

    # Group by Country and calculate mean salary
    data = df.groupby(["Country"])["Salary"].mean().sort_values(ascending=True)
    # Create a Plotly bar chart
    fig = px.bar(data, x=data.values, y=data.index, orientation='h') #, title='Mean Salary Based On Country')
    # Update layout if needed
    fig.update_layout(height=700, width=800) # set custom dimensions if required
    # Update the font size of the axis titles and the tick labels
    fig.update_layout(
        xaxis_title='Average Salary',
        yaxis_title='Country',
        xaxis=dict(
            title='Average Salary',
            titlefont_size=17,  # Change this value to adjust the title font size
            tickfont_size=15,   # Change this value to adjust the tick labels font size
        ),
        yaxis=dict(
            title='Country',
            titlefont_size=17,  # Change this value to adjust the title font size
            tickfont_size=15,   # Change this value to adjust the tick labels font size
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    
    
    
    st.write("#### Mean Salary Based On Experience in South Americ")

    # Group by YearsCodePro and calculate mean salary, ensure the index is sorted to avoid zigzag lines
    data = df.groupby(["Years of Experience"])["Salary"].mean().reset_index()
    data = data.sort_values("Years of Experience")

    # Create a Plotly line chart
    fig = px.line(data, x="Years of Experience", y="Salary")

    # Update the font size of the axis titles and the tick labels
    fig.update_layout(
        xaxis=dict(
            title='Years of Professional Coding Experience',
            titlefont_size=17,  # Adjust the title font size
            tickfont_size=15,   # Adjust the tick labels font size
        ),
        yaxis=dict(
            title='Average Salary',
            titlefont_size=17,  # Adjust the title font size
            tickfont_size=15,   # Adjust the tick labels font size
        )
    )

    # Display the line chart using Streamlit
    st.plotly_chart(fig, use_container_width=True)
    


##########

    # Plotting the heatmap with Plotly
    st.write("#### Correlation Heatmap of features for Salary Prediction in South America")
         
    # Load your data
    south_america_df = pd.read_csv('south_america.csv')

    features = ['Country', 'Education Level', 'Years of Experience', 'Salary']

    # Create a subset of the DataFrame with the features of interest
    df_features = south_america_df[features]

    # Apply OneHotEncoder to the categorical features
    ohe = OneHotEncoder(sparse=False)
    encoded_categorical = ohe.fit_transform(df_features[['Country', 'Education Level']])

    # Create a DataFrame with the encoded variables
    encoded_df = pd.DataFrame(encoded_categorical, columns=ohe.get_feature_names_out(['Country', 'Education Level']))

    # Add the numerical feature(s) and target variable to the encoded DataFrame
    encoded_df['Years of Experience'] = df_features['Years of Experience']
    encoded_df['Salary'] = df_features['Salary']

    # Calculate the correlation matrix
    correlation_matrix = encoded_df.corr()

    # Create the heatmap with text
    trace = go.Heatmap(
        z=correlation_matrix.values,
        x=correlation_matrix.columns.tolist(),
        y=correlation_matrix.columns.tolist(),
        colorscale='Blues',
        text=correlation_matrix.round(2).astype(str).values,
        texttemplate="%{text}",
        hoverinfo="none"  # This disables the hoverinfo if you want a cleaner look
    )

    # Layout for the plot
    layout = go.Layout(
        xaxis=dict(tickangle=-45),
        yaxis=dict(tickmode='array', tickvals=list(range(len(correlation_matrix.columns.tolist()))), ticktext=correlation_matrix.columns.tolist()),
        autosize=False,
        width=800,
        height=600,
    )

    # Combine the trace and layout into a figure
    fig = go.Figure(data=[trace], layout=layout)

    # Use Streamlit to display the Plotly figure
    st.plotly_chart(fig)
