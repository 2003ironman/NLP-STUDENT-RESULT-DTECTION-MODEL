import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Define file paths for the datasets
mat_file_path = r"C:\Users\Aditya Singh\Desktop\nlp assignment\student-mat.csv"
por_file_path = r"C:\Users\Aditya Singh\Desktop\nlp assignment\student-por.csv"
txt_file_path = r"C:\Users\Aditya Singh\Desktop\nlp assignment\student.txt"

# Load the datasets
student_mat_data = pd.read_csv(mat_file_path, sep=';')
student_por_data = pd.read_csv(por_file_path, sep=';')

# Function to perform analyses and generate visualizations
def analyze_student_data(student_data, title_suffix):
    # Calculate summary statistics
    summary_statistics = student_data.describe()
    print(f"Summary Statistics for {title_suffix}:\n", summary_statistics)

    # Select only numerical columns for correlation matrix
    numerical_columns = student_data.select_dtypes(include=['number']).columns
    correlation_matrix = student_data[numerical_columns].corr()

    # Plot the correlation matrix
    plt.figure(figsize=(14, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title(f'Correlation Matrix of Student Performance Data ({title_suffix})')
    plt.show()

    # Generate insights based on correlation matrix
    generate_insights_from_correlation(correlation_matrix, title_suffix)

    # Plot the distribution of final grades (G3)
    plt.figure(figsize=(10, 6))
    sns.histplot(student_data['G3'], bins=20, kde=True)
    plt.title(f'Distribution of Final Grades (G3) ({title_suffix})')
    plt.xlabel('Final Grade (G3)')
    plt.ylabel('Frequency')
    plt.show()

    # Plot the impact of study time on final grades (G3)
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='studytime', y='G3', data=student_data)
    plt.title(f'Impact of Study Time on Final Grades (G3) ({title_suffix})')
    plt.xlabel('Study Time')
    plt.ylabel('Final Grade (G3)')
    plt.show()

    # Plot the impact of weekday alcohol consumption (Dalc) on final grades (G3)
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Dalc', y='G3', data=student_data)
    plt.title(f'Impact of Weekday Alcohol Consumption on Final Grades (G3) ({title_suffix})')
    plt.xlabel('Weekday Alcohol Consumption (Dalc)')
    plt.ylabel('Final Grade (G3)')
    plt.show()

    # Plot the impact of weekend alcohol consumption (Walc) on final grades (G3)
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Walc', y='G3', data=student_data)
    plt.title(f'Impact of Weekend Alcohol Consumption on Final Grades (G3) ({title_suffix})')
    plt.xlabel('Weekend Alcohol Consumption (Walc)')
    plt.ylabel('Final Grade (G3)')
    plt.show()

    # Generate additional insights
    generate_insights(student_data, title_suffix)

    # Preprocess the data for machine learning
    X = student_data.drop('G3', axis=1).select_dtypes(include=[np.number])
    y = student_data['G3']

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions
    predictions = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    print(f"Mean Squared Error for {title_suffix}: {mse}")
    print(f"R^2 Score for {title_suffix}: {r2}")

    # Plot the true vs predicted values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, predictions)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
    plt.title(f'True vs Predicted Final Grades (G3) ({title_suffix})')
    plt.xlabel('True Grades (G3)')
    plt.ylabel('Predicted Grades (G3)')
    plt.show()

# Function to generate insights from the correlation matrix
def generate_insights_from_correlation(correlation_matrix, title_suffix):
    strong_correlations = correlation_matrix[(correlation_matrix > 0.5) | (correlation_matrix < -0.5)]
    strong_correlations = strong_correlations.dropna(how='all', axis=0).dropna(how='all', axis=1)
    print(f"Strong Correlations in {title_suffix} Data:\n", strong_correlations)

    # Generate some heuristic-based insights
    warnings = []
    suggestions = []
    positives = []
    negatives = []

    for col in strong_correlations.columns:
        for idx in strong_correlations.index:
            if col != idx and not np.isnan(strong_correlations.loc[idx, col]):
                corr_value = strong_correlations.loc[idx, col]
                if corr_value > 0.7:
                    positives.append(f"High positive correlation ({corr_value:.2f}) between {col} and {idx}.")
                elif corr_value < -0.7:
                    negatives.append(f"High negative correlation ({corr_value:.2f}) between {col} and {idx}.")
                elif corr_value > 0.5:
                    suggestions.append(f"Moderate positive correlation ({corr_value:.2f}) between {col} and {idx}.")
                elif corr_value < -0.5:
                    warnings.append(f"Moderate negative correlation ({corr_value:.2f}) between {col} and {idx}.")

    # Print insights
    print(f"\nWarnings for {title_suffix} Data:")
    for warning in warnings:
        print(warning)

    print(f"\nSuggestions for {title_suffix} Data:")
    for suggestion in suggestions:
        print(suggestion)

    print(f"\nPositives for {title_suffix} Data:")
    for positive in positives:
        print(positive)

    print(f"\nNegatives for {title_suffix} Data:")
    for negative in negatives:
        print(negative)

# Function to generate additional insights
def generate_insights(student_data, title_suffix):
    # Generate some heuristic-based insights
    warnings = []
    suggestions = []
    positives = []
    negatives = []

    # Impact of study time on final grades
    avg_grades_by_studytime = student_data.groupby('studytime')['G3'].mean()
    if avg_grades_by_studytime.idxmax() > 2:
        positives.append("Students who study more than 2 hours tend to have higher final grades.")
    else:
        suggestions.append("Encourage students to increase study time to improve final grades.")

    # Impact of weekday alcohol consumption on final grades
    avg_grades_by_dalc = student_data.groupby('Dalc')['G3'].mean()
    if avg_grades_by_dalc.idxmax() > 2:
        warnings.append("High weekday alcohol consumption is associated with lower final grades.")
    else:
        positives.append("Low weekday alcohol consumption is associated with higher final grades.")

    # Impact of weekend alcohol consumption on final grades
    avg_grades_by_walc = student_data.groupby('Walc')['G3'].mean()
    if avg_grades_by_walc.idxmax() > 2:
        warnings.append("High weekend alcohol consumption is associated with lower final grades.")
    else:
        positives.append("Low weekend alcohol consumption is associated with higher final grades.")

    # Print insights
    print(f"\nAdditional Warnings for {title_suffix} Data:")
    for warning in warnings:
        print(warning)

    print(f"\nAdditional Suggestions for {title_suffix} Data:")
    for suggestion in suggestions:
        print(suggestion)

    print(f"\nAdditional Positives for {title_suffix} Data:")
    for positive in positives:
        print(positive)

    print(f"\nAdditional Negatives for {title_suffix} Data:")
    for negative in negatives:
        print(negative)

# Analyze the student performance data for mathematics
analyze_student_data(student_mat_data, "Mathematics")

# Analyze the student performance data for Portuguese
analyze_student_data(student_por_data, "Portuguese")
