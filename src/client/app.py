import streamlit as st
import pandas as pd
import logging
from client import Client

# Configure logging to save all logs to a file and console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    handlers=[logging.FileHandler('client.log', encoding='utf-8')])

logger = logging.getLogger(__name__)

# Streamlit application for interacting with the Naive Bayes backend
class App:
    def __init__(self):
        self.__client = Client()
        if 'model_created' not in st.session_state:
            st.session_state.model_created = False
        if 'df' not in st.session_state:
            st.session_state.df = None
        if 'target_column' not in st.session_state:
            st.session_state.target_column = None
        if 'feature_columns' not in st.session_state:
            st.session_state.feature_columns = []

    # Load CSV data from user upload and select target column
    @staticmethod
    def load_data():
        st.header("Create model")
        uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state.df = df
                logger.info("File uploaded and loaded successfully.")
                st.success("File loaded successfully.")
                with st.expander("Preview the data"):
                    st.dataframe(df.head())
                target_column = st.selectbox("Select the target column", options=df.columns.tolist())
                if target_column:
                    st.session_state.target_column = target_column
                    st.session_state.feature_columns = [col for col in df.columns if col != target_column]
                    return df, target_column
            except Exception as e:
                logger.error(f"Error processing the uploaded file: {e}")
                st.error(f"Error processing the file: {str(e)}")
        return None, None

    # Send data and target column to backend to create a model
    def create_model(self, df, target_column):
        if st.button("create model", type="primary"):
            with st.spinner("creating model"):
                logger.info("Sending data to backend to create model.")
                result = self.__client.create_model(df, target_column)
            if "Error" in result:
                logger.error(f"Error creating the model: {result['Error']}")
                st.error(f"Error creating the model: {result['Error']}")
            else:
                logger.info("The model was successfully created!")
                st.success("The model was successfully created!")
                st.session_state.model_created = True

    # Collect user input for prediction based on feature columns
    @staticmethod
    def get_user_input():
        user_input = {}
        cols = st.columns(2)
        for i, feature in enumerate(st.session_state.feature_columns):
            col = cols[i % 2]
            with col:
                if st.session_state.df[feature].dtype in ['int64', 'float64']:
                    user_input[feature] = st.number_input(f"{feature}")
                else:
                    unique_values = st.session_state.df[feature].unique().tolist()
                    user_input[feature] = st.selectbox(f"{feature}", options=unique_values)
        return user_input

    # Display prediction interface and results
    def prediction(self):
        st.header("prediction")
        if st.session_state.model_created and st.session_state.df is not None:
            st.success("The model is ready for predictions!")
            st.subheader("Enter data to predict")

            user_input = self.get_user_input()
            if st.button("predict", type="primary"):
                with st.spinner("Predicting..."):
                    logger.info(f"Sending prediction request")
                    prediction_result = self.__client.get_prediction(user_input)

                if "Error" in prediction_result:
                    logger.error(f"Error predicting: {prediction_result['Error']}")
                    st.error(f"Error predicting: {prediction_result['Error']}")
                else:
                    st.success(f"prediction: {prediction_result['prediction']}")
        else:
            st.info("A model must be created first to make predictions")

    # Display model accuracy from backend
    def accuracy(self):
        st.header("The accuracy of the model")
        if st.session_state.model_created:
            if st.button("Check model accuracy"):
                with st.spinner("Checking model accuracy..."):
                    logger.info("Requesting model accuracy from backend.")
                    precision_result = self.__client.get_precision()
                if "Error" in precision_result:
                    logger.error(f"Error getting model accuracy: {precision_result['Error']}")
                    st.error(f"Error getting model accuracy: {precision_result['Error']}")
                else:
                    st.metric("Accuracy of the current model", f"{precision_result['model precision']:.2f}%")
        else:
            st.info("Create a model first to check accuracy")

    def run(self):
        st.set_page_config(page_title="Naive Bayes Model Client")
        st.title("Naive Bayes client")
        data, target = self.load_data()
        st.markdown("---")
        self.create_model(data, target)
        st.markdown("---")
        self.prediction()
        st.markdown("---")
        self.accuracy()


if __name__ == "__main__":
    app = App()
    app.run()



