import streamlit as st
from predict_EU import show_predict_page_eu
from explore_page import show_explore_page
from predict_South import show_predict_page_south
from predict_North import show_predict_page_north
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


# Load datasets
@st.cache_data
def load_data(file):
    data = pd.read_csv(file)
    return data

europe_df = load_data('europe.csv')
south_america_df = load_data('south_america.csv')
north_america_df = load_data('north_america.csv')

# page = st.sidebar.selectbox("Explore Or Predict", ("Home", "Europe", "South America", "North America"))

# Sidebar Navigation using a select box
st.sidebar.title("Navigation Bar")
page = st.sidebar.selectbox("Explore and Predict Salaries in NA - SA - EU", ["🏠 Home", "🌎 South America", "🌍 Europe", "🌎 North America"])

# Handling the page selection
if page == "🏠 Home":
    show_explore_page()
elif page == "🌎 South America":
    show_predict_page_south(),
elif page == "🌎 North America":
    show_predict_page_north()
else:  # Assuming the default is Europe if none of the above is selected
    show_predict_page_eu()

# Sidebar content
with st.sidebar:
    with st.expander("More information about this project"):
        st.markdown("""
        - **Project**: Exploration of the Software Engineer Salary Landscape in Europe - South America - North America
        - **Authors of the App**:
            - Fausto Bravo Cuvi
            - Karen Hourican
        - **Assignment**: DATA SCIENCE CLASS
        - **Teacher**: Miguel Sanz
        """)


st.markdown(""" """)
# Sidebar content

def plot_salary_by_education(region_df, title, palette):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Education Level', y='Salary', data=region_df, ci=None, palette=palette)
    plt.title(title)
    plt.xlabel('Education Level')
    plt.ylabel('Average Salary')
    plt.xticks(rotation=45)
    st.pyplot(fig)

def plot_contour_salary_experience(region_df, title, cmap):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.kdeplot(x='Years of Experience', y='Salary', data=region_df, cmap=cmap, fill=True)
    plt.title(title)
    plt.xlabel('Years of Experience')
    plt.ylabel('Salary')
    st.pyplot(fig)

with st.sidebar:

    st.markdown("""
    <div style="text-align: center">
                
    # 🧐 Visualize expected Average Salary by Continent considering:  
                
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div style="text-align: justify">

        - 👨🏻‍🎓 Education Level
        - 🧑🏻‍💻..🧓🏽 Years of Experience
                
        </div>
        """, unsafe_allow_html=True)
    # Use to navigate between different graphs
        
    graph = st.selectbox("Choose a graph type (Histogram by Education Level or Contour Plot Salary vs. Experience)", ["Histogram by Education Level", "Contour Plot Salary vs. Experience"])

    # Depending on the graph chosen, show the appropriate plots
    if graph == "Histogram by Education Level":
        st.write("**Choose a continent** 📍🗺️")
        continent = st.radio("Continent", ["Europe", "South America", "North America"])
        if continent == "Europe":
            plot_salary_by_education(europe_df, 'Average Salary by Education Level in Europe', 'Blues')
        elif continent == "South America":
            plot_salary_by_education(south_america_df, 'Average Salary by Education Level in South America', 'Greens')
        elif continent == "North America":
            plot_salary_by_education(north_america_df, 'Average Salary by Education Level in North America', 'Reds')
    elif graph == "Contour Plot Salary vs. Experience":
        st.write("**Choose a continent** 📍🗺️")
        continent = st.radio("Continent", ["Europe", "South America", "North America"])
        if continent == "Europe":
            plot_contour_salary_experience(europe_df, 'Contour Plot for Salary vs. Years of Experience in Europe', 'Blues')
        elif continent == "South America":
            plot_contour_salary_experience(south_america_df, 'Contour Plot for Salary vs. Years of Experience in South America', 'Greens')
        elif continent == "North America":
            plot_contour_salary_experience(north_america_df, 'Contour Plot for Salary vs. Years of Experience in North America', 'Reds')



