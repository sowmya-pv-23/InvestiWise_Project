import streamlit as st
from streamlit_option_menu import option_menu
import pickle
import numpy as np
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
import os
import pandas as pd
import plotly.graph_objects as go



st.set_page_config(page_title="InvestiWise",
                   layout="wide",
                   page_icon=" ")
# placeholder = st.image("/content/images (1).jpg")




# working_dir = os.path.dirname(os.path.abspath(__file__))
# rule_based_path = os.path.join(working_dir, 'rule_based.py')
# from rule_based import predict_investment_risk
# st.image(r"C:\Users\sowmy\Downloads\istockphoto-1297492947-612x612.jpg")
# st.sidebar.title("Enter the credentials")
# st.sidebar.text_input("Enter user name")
# st.sidebar.text_input("Password")
# st.sidebar.button("login")
# credit_rate = pickle.load(open(f'{working_dir}/models/Credit_rating _model.pkl', 'rb'))
# scaler = pickle.load(open(f'{working_dir}/models/min_max_scaler.pkl', 'rb'))
# model_path = os.path.join(working_dir, 'models/saved_model')
# tokenizer_path = os.path.join(working_dir, 'models/DistilBert_Tokenizer')
# df = pd.read_csv(os.path.join(working_dir, 'Datasets/Visual_ESG_DATASET.csv'))
# with st.sidebar:
#     selected = option_menu("Comprehensive Investment Risk Analysis",
#                            ['InvestiWise:',
#                             'Investment Risk Prediction',
#                             'Data Viewer',
#                             'Performance Analysis'],
#                            icons=['', 'graph-up-arrow', 'file-text', 'bar-chart'],
#                            default_index=0
#                            )
def home():
    st.title("InvestiWise")
    st.markdown("<h3 style='color: green;'>A sustainable investment dashboard</h3>", unsafe_allow_html=True)
    placeholder = st.image("/content/Screenshot 2024-06-26 183437.png")


def investment_risk_prediction():
    placeholder=st.empty()
    working_dir = os.path.dirname(os.path.abspath(__file__))
    rule_based_path = os.path.join(working_dir, 'rule_based.py')
    from rule_based import predict_investment_risk
    credit_rate = pickle.load(open(f'{working_dir}/Models/Credit_rating _model.pkl', 'rb'))
    scaler = pickle.load(open(f'{working_dir}/Models/min_max_scaler.pkl', 'rb'))
    model_path = os.path.join(working_dir, 'Models/saved_model')
    tokenizer_path = os.path.join(working_dir, 'Models/DistilBert_Tokenizer')
    
    st.title('Investment Risk Prediction using ML')
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        Market = st.selectbox("Select Market", ['Emerging Markets', 'Developed Markets'])
    with col3:
        Region = st.selectbox("Select Region", ['Americas', 'Asia', 'CEEMEA', 'Europe'])
    if Market == 'Emerging Markets':
        if Region == 'Americas':
            Country_risk_market_return = st.slider('Country Risk Market Return', 0.0, 30.00)
            Country_risk_premium = st.slider('Country Risk Premium', 0.0, 15.0)
            Country_risk_rfr = st.slider('Country Risk free Rate', 0.0, 10.00)
            Risk_Premium = st.slider('Risk Premium', 0.00, 35.00)
        elif Region == 'Asia':
            Country_risk_market_return = st.slider('Country Risk Market Return', 0.0, 15.00)
            Country_risk_premium = st.slider('Country Risk Premium', 0.0, 15.0)
            Country_risk_rfr = st.slider('Country Risk free Rate', 0.0, 10.00)
            Risk_Premium = st.slider('Risk Premium', -1.00, 20.00)
        elif Region == 'CEEMEA':
            Country_risk_market_return = st.slider('Country Risk Market Return', 0.0, 20.00)
            Country_risk_premium = st.slider('Country Risk Premium', 0.0, 15.0)
            Country_risk_rfr = st.slider('Country Risk free Rate', 0.0, 15.00)
            Risk_Premium = st.slider('Risk Premium', 0.00, 15.00)
        else:
            st.warning('Change the Market type to Developed Markets')
    if Market == 'Developed Markets':
        if Region == 'Americas':
            Country_risk_market_return = st.slider('Country Risk Market Return', 0.0, 15.00)
            Country_risk_premium = st.slider('Country Risk Premium', 0.0, 15.0)
            Country_risk_rfr = st.slider('Country Risk free Rate', 0.0, 10.00)
            Risk_Premium = st.slider('Risk Premium', 0.00, 20.00)
        elif Region == 'Asia':
            Country_risk_market_return = st.slider('Country Risk Market Return', 0.0, 15.00)
            Country_risk_premium = st.slider('Country Risk Premium', 0.0, 15.00)
            Country_risk_rfr = st.slider('Country Risk free Rate', -1.00, 3.00)
            Risk_Premium = st.slider('Risk Premium', 0.00, 20.00)
        elif Region == 'Europe':
            Country_risk_market_return = st.slider('Country Risk Market Return', 0.0, 20.00)
            Country_risk_premium = st.slider('Country Risk Premium', 0.0, 15.0)
            Country_risk_rfr = st.slider('Country Risk free Rate', -1.00, 5.00)
            Risk_Premium = st.slider('Risk Premium', 0.00, 30.00)
        else:
            st.warning('Change the Market type to Emerging Markets')

    Gross_Margin = st.slider('Gross Margin', 0.00, 100.00)
    Is_int_EXP = st.slider('Interest Expense', 0.00, 8500.00, )
    Oper_Margin = st.slider('Operating Margin', -50.00, 100.00)
    Unlevered_Beta = st.slider('Unlevered Beta', -2.00, 3.50)
    WACC = st.slider('Weighted Average Cost of Capital(WACC)', 0.00, 25.00)
    WACC_COST_DEBT = st.slider('WACC Cost Debt', 0.00, 10.00)
    WACC_COST_Equity = st.slider('WACC Cost Equity', 0.00, 20.00)
    EPS_Growth = st.slider('Earning Per Share Growth', -600, 900)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        Total_E = st.number_input('Environment Score', 0, 20)
    with col2:
        Total_S = st.number_input('Social Score', 0, 20)
    with col3:
        Total_G = st.number_input('Governance Score', 0, 20)
    news = st.text_input('News')
    invest_pred = ''
    col1, col2, col3, col4 = st.columns([1, 2, 3, 1])
    with col3:
        predictions = st.button('Predict')

    if predictions:
        
       
        if Market == 'Emerging Market':
            market_input = 1
        else:
            market_input = 0

        # User input for Region (assuming only one region can be selected)
        region_input = {
            'Asia': 1 if Region == 'Asia' else 0,
            'CEEMEA': 1 if Region == 'CEEMEA' else 0,
            'Europe': 1 if Region == 'Europe' else 0
        }
        numerical_inputs = [Country_risk_market_return, Country_risk_premium,
                            Country_risk_rfr, EPS_Growth, Gross_Margin,
                            Is_int_EXP, Oper_Margin, Risk_Premium,
                            Unlevered_Beta, WACC, WACC_COST_DEBT, WACC_COST_Equity, Total_E, Total_S, Total_G]
        # categorical_data=[Market,Region]
        # Assuming Market and Region are categorical variables
        market_input_array = np.array([market_input]).reshape(1, -1)
        region_input_array = np.array([list(region_input.values())]).reshape(1, -1)
        # numerical_inputs_array = np.array(numerical_inputs).reshape(1, -1)
        numerical_inputs_array = np.array(numerical_inputs).reshape(1, -1)
     
        input_vector = np.concatenate((numerical_inputs_array, market_input_array, region_input_array), axis=1)
        scaled_inputs = scaler.transform(input_vector)

        # st.warning(input_vector)
        # Predict using the model
        credit_rating_impact = credit_rate.predict(scaled_inputs)

        news_input = [news]

        tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_path)

        model = DistilBertForSequenceClassification.from_pretrained(model_path, from_tf=True)
        inputs = tokenizer(news, return_tensors="pt", truncation=True, padding=True, max_length=512)

        # Perform the prediction
        with torch.no_grad():
            outputs = model(**inputs)

        # Get the predicted class
        predicted_class = torch.argmax(outputs.logits, dim=1).item()

        # Map class to sentiment
        sentiment_map = {0: -1, 1: 0, 2: 1}
        predicted_sentiment = sentiment_map[predicted_class]
        # combined_ESG = Total_E + Total_S + Total_G
    
        
        investment_risk=predict_investment_risk(credit_rating_impact,Total_E,Total_S,Total_G,predicted_sentiment)
        st.write(investment_risk)

    st.success(invest_pred)
def data_viewer():
    
    placeholder=st.empty()
    st.title('Detailed View')
    working_dir = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv(os.path.join(working_dir, 'Datasets/Visual_ESG_DATASET.csv'))
    df_subset = df.sample(n=1000, random_state=42)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        company = st.selectbox('Select Company (optional)', [' '] + list(df_subset['Company'].unique()))
        
    if company != ' ':
        filtered_df = df_subset[df_subset['Company'] == company]
    else:
        with col2:
            market = st.multiselect('Select Market', df_subset['Market'].unique())
        with col3:
            sector = st.multiselect('Select Sector', df_subset['Sector'].unique())
        
        if market and sector:
            filtered_df = df_subset[
                (df_subset['Market'].isin(market)) &
                (df_subset['Sector'].isin(sector))
            ]
        elif market:
            filtered_df = df_subset[df_subset['Market'].isin(market)]
        elif sector:
            filtered_df = df_subset[df_subset['Sector'].isin(sector)]
        else:
            filtered_df = pd.DataFrame()  # Initialize as empty DataFrame

    if not filtered_df.empty:
        st.write(filtered_df[['Company', 'Region', 'Market', 'Sector', 'COUNTRY_RISK_MARKET_RETURN', 
                              'COUNTRY_RISK_RFR', 'COUNTRY_RISK_PREMIUM', 'GROSS_MARGIN', 'OPER_MARGIN', 'EPS_GROWTH',
                              'UNLEVERED_BETA', 'WACC', 'Credit rating impact', 'Total E', 'Total S', 'Total G']].set_index('Company'))
    else:
        st.write("No data available for the selected filters.")

   
    # else:
    #     st.write("No data available for the selected market(s).")
    
    # st.dataframe(df_subset, use_container_width=True)
def performance_analysis():
    working_dir = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv(os.path.join(working_dir, 'Datasets/Visual_ESG_DATASET.csv'))
    df = df.sample(n=1000, random_state=42)
    placeholder=st.empty()
    st.title('Performance Analysis')
    companies = st.multiselect('Select Companies (up to two)', list(df['Company'].unique()))
    # companies = st.multiselect('Select two companies to compare', df['Company'].unique())

    if len(companies) == 2:
        company1, company2 = companies

        # Get data for the selected companies
        data = df[df['Company'].isin(companies)].set_index('Company')

        # Define financial metrics to compare
        financial_metrics = ['COUNTRY_RISK_MARKET_RETURN', 'COUNTRY_RISK_RFR', 'COUNTRY_RISK_PREMIUM',
                             'GROSS_MARGIN', 'OPER_MARGIN', 'EPS_GROWTH', 'UNLEVERED_BETA', 'WACC', 'Credit rating impact']
        
        # Define ESG metrics to compare
        esg_metrics = ['Total E', 'Total S', 'Total G']
        st.subheader("Financial Metrics - Horizontal Bar Chart")
        fig1 = go.Figure()
        
        # st.subheader("Financial Metrics - Horizontal Bar Chart")
        # fig1 = go.Figure()
        
        # Add bars for company1
        fig1.add_trace(go.Bar(
            y=financial_metrics, 
            x=data.loc[company1, financial_metrics], 
            orientation='h',  # Horizontal bar chart
            name=company1,
            marker_color='blue',
            marker_line=dict(color='black', width=1.5),
        ))
        
        # Add bars for company2
        fig1.add_trace(go.Bar(
            y=financial_metrics, 
            x=data.loc[company2, financial_metrics], 
            orientation='h',  # Horizontal bar chart
            name=company2,
            marker_color='orange',
            marker_line=dict(color='black', width=1.5),
        ))
        
        fig1.update_layout(
            barmode='group',  # Group bars
            yaxis_title="Metrics",
            xaxis_title="Values",
            height=400,
            width=800,
            # legend=dict(title="Companies", orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig1)
        
        st.subheader("ESG Metrics - Grouped Bar Chart")
        fig2 = go.Figure()
        
        # Add bars for company1
        fig2.add_trace(go.Bar(
            x=esg_metrics, 
            y=data.loc[company1, esg_metrics], 
            name=company1,
            marker_color='blue',
            marker_line=dict(color='black', width=1.5),
        ))
        
        # Add bars for company2
        fig2.add_trace(go.Bar(
            x=esg_metrics, 
            y=data.loc[company2, esg_metrics], 
            name=company2,
            marker_color='orange',
            marker_line=dict(color='black', width=1.5),
        ))
        
        fig2.update_layout(
            barmode='group',  # Group bars
            xaxis_title="Metrics",
            yaxis_title="Values",
            height=600,
            width=800,
            # legend=dict(title="Companies", orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig2)


        



# st.set_page_config(page_title='Comprehensive Investment Risk Analysis', page_icon=':bar_chart:', layout='wide')

with st.sidebar:
    selected = option_menu(
        "Comprehensive Investment Risk Analysis",
        ["InvestiWise:", "Investment Risk Prediction", "Data Viewer", "Performance Analysis"],
        icons=["house", "graph-up-arrow", "table", "clipboard-data"],
        menu_icon="cast",
        default_index=0,
    )

if selected == "InvestiWise:":
    home()
elif selected == "Investment Risk Prediction":
    investment_risk_prediction()
elif selected == "Data Viewer":
    data_viewer()
elif selected == "Performance Analysis":
    performance_analysis()
   

    
       
    
           
    


