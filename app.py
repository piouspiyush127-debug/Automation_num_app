import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from statsmodels.stats.outliers_influence import variance_inflation_factor
import nltk


# Ensure nltk resources are downloaded
nltk.download('punkt')


# Function to load data
def load_data(uploaded_file):
    if uploaded_file.name.endswith('.csv'):
        return pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith('.json'):
        return pd.read_json(uploaded_file)
    elif uploaded_file.name.endswith('.xlsx'):
        return pd.read_excel(uploaded_file)
    else:
        st.error("Unsupported file type")
        return None


# Data Cleaning Function
def clean_data(df):
    st.write("### Data Cleaning")

    # Handling missing values
    st.write("Handling Missing Values:")
    missing_value_strategy = st.selectbox("Choose a strategy to handle missing values:",
                                          ["Forward Fill", "Backward Fill", "Mean", "Median", "Drop Rows"])

    if missing_value_strategy == "Forward Fill":
        df.fillna(method='ffill', inplace=True)
    elif missing_value_strategy == "Backward Fill":
        df.fillna(method='bfill', inplace=True)
    elif missing_value_strategy == "Mean":
        df.fillna(df.mean(), inplace=True)
    elif missing_value_strategy == "Median":
        df.fillna(df.median(), inplace=True)
    elif missing_value_strategy == "Drop Rows":
        df.dropna(inplace=True)

    # Handling duplicates
    if st.checkbox("Remove duplicates"):
        df.drop_duplicates(inplace=True)
        st.write("Duplicates removed.")

    # Handling outliers
    if st.checkbox("Remove outliers based on Z-score"):
        from scipy.stats import zscore
        z_scores = np.abs(zscore(df.select_dtypes(include=[np.number])))
        df = df[(z_scores < 3).all(axis=1)]
        st.write("Outliers removed.")

    # Dropping irrelevant columns
    columns_to_drop = st.multiselect("Select columns to drop:", df.columns)
    if columns_to_drop:
        df.drop(columns=columns_to_drop, inplace=True)
        st.write(f"Dropped columns: {columns_to_drop}")

    return df


# Data Transformation Function
def transform_data(df):
    st.write("### Data Transformation")

    # Scaling numerical features
    if st.checkbox("Standardize numerical features"):
        scaler = StandardScaler()
        num_cols = df.select_dtypes(include=[np.number]).columns
        df[num_cols] = scaler.fit_transform(df[num_cols])
        st.write("Numerical features standardized.")

    # Encoding categorical variables
    if st.checkbox("One-Hot Encode categorical features"):
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        df = pd.get_dummies(df, columns=cat_cols)
        st.write("Categorical features one-hot encoded.")

    return df


# Data Validation Function
def validate_data(df):
    st.write("### Data Validation")

    # Check for data type consistency
    if st.checkbox("Check for data type consistency"):
        st.write("Data Types:")
        st.write(df.dtypes)

    # Check for correlations
    if st.checkbox("Check for correlations"):
        corr_matrix = df.corr()
        st.write("Correlation Matrix:")
        st.write(corr_matrix)

        st.write("Heatmap of Correlations:")
        fig, ax = plt.subplots()
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    # Check for multicollinearity
    if st.checkbox("Check for multicollinearity"):
        vif_data = pd.DataFrame()
        vif_data["feature"] = df.columns
        vif_data["VIF"] = [variance_inflation_factor(df.values, i) for i in range(len(df.columns))]
        st.write("Variance Inflation Factor (VIF):")
        st.write(vif_data)

    return df


def evaluate_models(df, x_cols, y_col):
    st.write("### Model Evaluation")

    # Ensure all selected columns are numeric
    for col in x_cols:
        if not np.issubdtype(df[col].dtype, np.number):
            st.error(f"Selected column {col} must be numeric for regression.")
            return None

    x = df[x_cols].values
    y = df[y_col].values

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(),
        "Lasso Regression": Lasso(),
        "Support Vector Regression": SVR(),
        "Polynomial Regression (Degree 2)": make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
    }

    for model_name, model in models.items():
        model.fit(x, y)

    return models


def generate_report(analysis_type, df, models, x_cols, y_col):
    st.write(f"## {analysis_type} Report")

    if analysis_type == "Regression":
        x = df[x_cols].values  # Use the selected independent variables
        y = df[y_col].values  # Use the selected dependent variable

        results = []

        for model_name, model in models.items():
            st.write(f"### {model_name} Summary")

            # Predictions
            model.fit(x, y)
            y_pred = model.predict(x)

            # Plotting each feature against the target variable
            for i, col in enumerate(x_cols):
                fig, ax = plt.subplots()
                sns.scatterplot(x=df[col], y=y, ax=ax, label='Data Points')

                if model_name.startswith("Polynomial") and x.shape[1] == 1:
                    # Polynomial Regression with one feature
                    x_range = np.linspace(df[col].min(), df[col].max(), 100).reshape(-1, 1)
                    y_range = model.predict(x_range)
                    ax.plot(x_range, y_range, color='red', label='Polynomial Fit')
                else:
                    # For other models, plot the regression line or fit
                    ax.plot(df[col], y_pred, color='red', label='Regression Line')

                ax.set_xlabel(col)
                ax.set_ylabel(y_col)
                ax.set_title(f"{model_name} Plot for {col}")
                ax.legend()
                st.pyplot(fig)

            # Calculate errors
            mse = mean_squared_error(y, y_pred)
            mae = mean_absolute_error(y, y_pred)
            rmse = np.sqrt(mse)

            st.write(f"Mean Absolute Error (MAE): {mae:.4f}")
            st.write(f"Mean Squared Error (MSE): {mse:.4f}")
            st.write(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

            results.append((model_name, mae, mse, rmse))

        # Identify the best model
        best_model = min(results, key=lambda x: x[1])[0]
        st.write("### Best Model")
        st.write(f"The best model is **{best_model}** with the lowest MAE.")
    else:
        st.error("Unsupported analysis type for reporting.")

# Main function for the Streamlit app
def main():
    st.title("AI Employee: Data Analysis and Reporting")
    st.write("Welcome! Please upload your data files to get started.")

    uploaded_file = st.file_uploader("Choose a file", type=['csv', 'json', 'xlsx'])
    if uploaded_file is not None:
        data = load_data(uploaded_file)
        if data is not None:
            st.write("Raw Data Preview:")
            st.dataframe(data.head())

            # Data Cleaning
            cleaned_data = clean_data(data)
            st.write("Cleaned Data Preview:")
            st.dataframe(cleaned_data.head())

            # Data Transformation
            transformed_data = transform_data(cleaned_data)
            st.write("Transformed Data Preview:")
            st.dataframe(transformed_data.head())

            # Data Validation
            validated_data = validate_data(transformed_data)
            st.write("Validated Data Preview:")
            st.dataframe(validated_data.head())

            # Model Selection and Evaluation
            x_cols = st.multiselect("Select the independent variables (X):", validated_data.columns)
            y_col = st.selectbox("Select the dependent variable (Y):", validated_data.columns)

            if x_cols and y_col and st.button("Run Regression Models"):
                models = evaluate_models(validated_data, x_cols, y_col)
                generate_report("Regression", validated_data, models, x_cols, y_col)


if __name__ == "__main__":
    main()
