import streamlit as st
from joblib import load
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder


def clean_experience(x):
    if x ==  'More than 50 years':
        return 50
    if x == 'Less than 1 year':
        return 0.5
    return float(x)

# Load the pipeline model
@st.cache_data
def load_model():
    pipeline = load('best_model_EU.joblib')
    return pipeline

pipeline_NA = load_model()

# Define the Streamlit page
def show_predict_page_eu():
    st.markdown("<h1 style='text-align: center; color: red;'>Salary Prediction in Europe for Software Engineers</h1>", unsafe_allow_html=True)
    st.write("""We need some information to predict the salary""")

    countries = (
        'Germany',
        'France',
        'Poland',
        'Netherlands',
        'Spain',
        'Italy',
        'Sweden',
        'United Kingdom',
        'Switzerland',
        'Austria',
        'Norway',
        'Denmark',
        'Belgium',
        'Ukraine',
        'Portugal',
        'Romania',
        'Finland',
        'Greece',
        'Hungary',
        'Ireland',
        'Bulgaria',
        'Serbia',
        'Slovenia',
        'Lithuania',
        'Slovakia',
        'Croatia',
        'Estonia',
        'Belarus',
        'Latvia',
        'Bosnia and Herzegovina',
        'Luxembourg',
        'Iceland',
        'Malta',
        'Albania',
        # 'Montenegro',
        # 'Andorra',
        # 'Monaco',
        # 'San Marino',
        # 'Liechtenstein'
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

# Run the prediction page function


    def load_data():
        df = pd.read_csv("europe.csv")
        df = df[["Country", "Education Level", "Years of Experience", "Salary"]]
        return df


    df = load_data()


    st.write("""""")
    st.write("""""")
    st.write("""""")

    st.markdown("<h2 style='text-align: center;';>Continue exploring about the salary landscape in Europe for Software Engineers</h2>", unsafe_allow_html=True)


    st.write("""""")
    st.write("""""")
    st.write("""""")
    st.write("""""")
    st.write("""""")
    
    
    st.write("""#### % of the Data surveyed from different countries in Europe""")
    # st.pyplot(fig1)
    data = df["Country"].value_counts()
    fig = px.pie(data, values=data.values, names=data.index)
    fig.update_traces(textposition='inside', textinfo='percent+label')
    # Set the height of the chart, this will make the pie chart bigger. 
    fig.update_layout(height=800)
    st.plotly_chart(fig, use_container_width=True)
    
    
    st.write("#### Mean Salary Based On Country in Europe")

    # Group by Country and calculate mean salary
    data = df.groupby(["Country"])["Salary"].mean().sort_values(ascending=True)
    # Create a Plotly bar chart
    fig = px.bar(data, x=data.values, y=data.index, orientation='h') 
    # Update layout if needed
    fig.update_layout(height=700, width=800) #
    # Update the font size of the axis titles and the tick labels
    fig.update_layout(
        xaxis_title='Average Salary',
        yaxis_title='Country',
        xaxis=dict(
            title='Average Salary',
            titlefont_size=17,  
            tickfont_size=15,   
        ),
        yaxis=dict(
            title='Country',
            titlefont_size=17,  
            tickfont_size=15,   
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    
    
    
    st.write("#### Mean Salary Based On Experience in Europe")

    # Group by YearsCodePro and calculate mean salary, ensure the index is sorted to avoid zigzag lines
    data = df.groupby(["Years of Experience"])["Salary"].mean().reset_index()
    data = data.sort_values("Years of Experience")

    # Create a Plotly line chart
    fig = px.line(data, x="Years of Experience", y="Salary")

    # Update the font size of the axis titles and the tick labels
    fig.update_layout(
        xaxis=dict(
            title='Years of Professional Coding Experience',
            titlefont_size=17,  
            tickfont_size=15,   
        ),
        yaxis=dict(
            title='Average Salary',
            titlefont_size=17,  
            tickfont_size=15,   
        )
    )

    # Display the line chart using Streamlit
    st.plotly_chart(fig, use_container_width=True)
    


    st.write("#### Correlation Heatmap of features for Salary Prediction in Europe")
    
    st.write("""""")
    st.write("""""")
    st.write("""""")

    # Load the data
    europe_df = pd.read_csv('europe.csv')

    # Select the top N countries based on the number of entries
    top_n_countries = europe_df['Country'].value_counts().nlargest(10).index

    # Filter the DataFrame to include only the top N countries
    europe_df = europe_df[europe_df['Country'].isin(top_n_countries)]

    features = ['Country', 'Education Level', 'Years of Experience', 'Salary']

    # Create a subset of the DataFrame with the features of interest
    df_features = europe_df[features]

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

    fig, ax = plt.subplots(figsize=(12, 8)) # This creates a new figure
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='Blues', ax=ax)
    plt.title('Correlation of features for Salary Prediction in Europe')

    # Set the background color of the axis (where the heatmap is plotted)
    ax.set_facecolor('none')  # 'none' is equivalent to transparent

    # If you want the entire figure to be transparent:
    fig.patch.set_facecolor('none')

    st.pyplot(fig)