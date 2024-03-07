import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import streamlit as st
from joblib import load
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns
import pydeck as pdk
import geopandas as gpd
import geopandas.datasets
import json


@st.cache_data()
def show_explore_page():

        # Create three columns
    col1, col2, col3, col4, col5 = st.columns([1,1,3,1,1])

    # Use the middle column to display the image, achieving a centered effect
    with col3:
        st.image("logo3.png", width=250)
    # Title and Introduction
    # st.title('EXPLORATION OF THE SOFTWARE ENGINEER SALARIES LANDSCAPE IN EUROPE - SOUTH AMERICA - NORTH AMERICA')
    # st.markdown("<h1 style='text-align: center; color: red;'>EXPLORATION OF THE SOFTWARE ENGINEER SALARY LANDSCAPE IN EUROPE - SOUTH AMERICA - NORTH AMERICA</h1>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center; color: red;'>💸 Software Engineer Salary Landscape in Europe - South America - North America 💰</h1>", unsafe_allow_html=True)
    
    st.markdown(""" """)
        
    st.markdown(""" """)

    st.markdown(""" """)

    st.markdown("""
        <div style="text-align: justify">
        This interactive application provides an in-depth analysis of the software engineer salary landscape across Europe, South America, and North America.
        Explore various insights into salary distributions, the impact of education, experience, and country on salaries, and predictions based on comprehensive data analysis.
        
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown(""" """)
    st.markdown("""
        <div style="text-align: justify">

                
        **If you would like to jump directly to try out the salary prediction model, you can use the sidebar to navigate to the desired continent and country, and get started with the predictions:**
        - 🏛️ Europe
        - 🏝️ South America
        - 🗽 North America
                
        </div>
        """, unsafe_allow_html=True)
                

    st.markdown(""" """)
    # General Objective
    st.write("""
    ### General Objective 🗺️
    """)

    st.markdown("""
        <div style="text-align: justify">
        To conduct a comprehensive data-driven analysis of software engineer 
        salaries across Europe, South America, and North America, with the aim of understanding regional disparities and developing robust, region-specific predictive models. This analysis seeks to offer a granular understanding of the factors influencing salary variations and to ensure accurate salary predictions that are sensitive to the economic realities of each continent.
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown(""" """)
    st.markdown(""" """)
    # Display the Specific Objectives with expandable sections
    with st.expander("**Objective 1:** Granular Salary Analysis by Continent", expanded=False):
        st.write("""
        To perform a detailed analysis of software engineering salaries within each specified geographical context — Europe, South America, and North America — to understand regional compensation trends and variations.
        """)

    with st.expander("**Objective 2:** Comparative Regional Analysis", expanded=False):
        st.write("""
        To compare and contrast the salary data across the aforementioned continents, identifying patterns and discrepancies that may influence wage scales in each region.
        """)

    with st.expander("**Objective 3:** Targeted Model Training", expanded=False):
        st.write("""
        To develop predictive models that are trained on continent-specific data, ensuring that the models account for regional economic factors and are not skewed by disparities in salary ranges between high-wage and low-wage areas.
        """)

    with st.expander("**Objective 4:** Mitigation of Wage Disparity Impact", expanded=False):
        st.write("""
        To mitigate the impact of wage disparities when aggregating salary data by training separate models for each continent. This approach aims to prevent high-salary regions from disproportionately affecting the predictive accuracy for regions with lower average salaries.
        """)

    with st.expander("**Objective 5:** Enhanced Accuracy in Salary Predictions", expanded=False):
        st.write("""
        To enhance the accuracy of salary predictions by creating tailored models that take into account the economic and professional nuances of each studied region, thereby providing more relevant insights for stakeholders.
        """)

    st.markdown(""" """)
   

    st.markdown(""" """)

    

    st.write(
        """
    ### Data Cleaning and Preprocessing 🧹 🗑️"""
    )
    st.write(""" """)
    st.write("""
    In the initial phase of our project, we dedicated substantial effort to cleaning and preprocessing the data to establish a solid foundation for our subsequent analysis.
    """)
    with st.expander("Click here for more details on the Data cleaninig and preprocessing", expanded=False):
        st.write("""
        **1. Data Verification: Granular Salary Analysis by Continent**
        
        We meticulously checked the raw data for accuracy, ensuring that there were no errors or inconsistencies that could compromise the integrity of our findings.
        
        **2. Standardization of Column Names**
        
        Recognizing that consistent nomenclature is crucial for data manipulation and interpretation, we standardized the names of the columns across our datasets. This involved unifying naming conventions, correcting any typographical errors, and aligning similar fields under a common terminology.
        
        **3. Category Reduction**
        
        In fields with a multitude of unique values, we strategically reduced the number of categories to a manageable level. This was particularly important for categorical variables where numerous sub-categories could potentially dilute the statistical power of our analysis. By grouping related categories and eliminating sparsely populated ones, we aimed to maintain analytical clarity without sacrificing the granularity of the data.
        
        **4. Geographical Grouping**
        
        We grouped individual countries into their respective continents—Europe, South America, and North America. This not only streamlined the analysis by reducing the complexity inherent in dealing with numerous distinct nations but also enabled us to focus on regional trends and disparities which are more pertinent to our study's objectives.

             """
        )

    st.write(""" """)
    st.write(
        """
    ### Data Overview 🔎"""
    )
    
    st.markdown("""
        <div style="text-align: justify">
    The dataset includes responses from software engineers across the three continents, detailing salaries, years of experience, and education levels.
    Our initial step involved cleaning and preprocessing this data to ensure accuracy and relevance for our analysis.
   </div>
        """, unsafe_allow_html=True)
    st.markdown(""" """)
    with st.expander("More Details on the Dataset", expanded=False):
        st.write("""
        We included 04 datasets in our analysis, each containing information on software engineer salaries, experience, and education levels. The datasets were sourced from the reputable Stack Overflow Annual Developer Survey of the years 2020, 2021, 2022, 2023.

        The Stack Overflow Annual Developer Survey is a comprehensive survey that gathers information from thousands of developers across the globe. This survey covers a range of topics, including the highest level of formal education that developers have completed, how developers learn to code, the online resources they use, their professional experience, and much more.

        One of the key takeaways from the 2023 survey was the varied educational backgrounds of the respondents, ranging from primary/elementary school to professional degrees such as JD, MD, and Ph.D.​​. The survey highlights the diverse ways in which individuals learn to code, showing an increasing trend towards online resources and certifications, especially among younger age groups. Moreover, the survey points out the popular online platforms and resources that developers rely on to learn coding, which includes technical documentation and communities like Stack Overflow itself​​.

        In addition to educational insights, the survey also sheds light on professional aspects such as the types of roles developers are engaged in, coding as a hobby, and the varying levels of professional coding experience among different developer types. For example, the 2019 survey provided data on how developers' professional coding experience varies across different roles within the industry​​.

        These surveys are essential for understanding the current state of the developer workforce, their habits, preferences, and the trends within the software development industry. They provide a valuable resource for anyone interested in the technology field, from companies and hiring managers to educators and the developers themselves.

        For the latest and more detailed findings, you can always visit the Stack Overflow Developer Survey results on their website.   
        
        Check out the data source [link](https://insights.stackoverflow.com/survey) 👈

                 
     """)
    


    