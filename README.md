
# Bitcoin Momentum Strategy with Machine Learning

## Overview
This project implements a momentum-based trading strategy for Bitcoin using supervised machine learning models to predict the profitability of trades. The strategy focuses on identifying entry points where both daily and weekly momentums are aligned, and then employs various technical indicators to engineer features for the machine learning models.

## Data
- **Daily Bitcoin Data (2014-2023)**: Historical daily data including Open, High, Low, Close, and Volume prices.
- **Weekly Bitcoin Data**: Derived from daily data by resampling and aggregating on a weekly basis.

## Project Structure

### 1. Data Preprocessing
- **Data Loading**:  
  Read the daily Bitcoin data from a CSV file.  
  Convert the 'Date' column to datetime format and set it as the index.

- **Weekly Data Creation**:  
  Resample the daily data to create a weekly dataset, aggregating Open, High, Low, Close, and Volume accordingly.  
  Save the weekly dataset for later use.

### 2. Momentum Strategy Implementation

- **Momentum Calculation**:  
  Define a function to calculate momentum over a 14-day window by finding the difference between the current closing price and the closing price 14 periods earlier.

- **Entry Point Identification**:  
  Identify entry points (ET) where there's a trend change, and daily and weekly momentums are aligned.

- **Signal Generation**:  
  Generate trading signals based on the momentum calculations.  
  Assign positions for trading: long (1) or short (-1).

- **Filtering Aligned Momentum**:  
  Filter entry points to include only those where both daily and weekly momentums are either positive or negative.

### 3. Feature Engineering
- **Technical Indicators Calculation**:  
  For each entry point, calculate 18 technical indicators using the previous 18 periods of data. These include:

  - **Price Action Variables**:  
    - Variance and standard deviation of closing prices.  
    - Average percentage change in returns.  
    - Slopes of closing price trends.  
    - Relative Strength Index (RSI) and its variance and slope.

  - **Volume Variables**:  
    - Variance and average percentage change of volume.  
    - Volume slope.

  - **Divergence Variables**:  
    - Correlations between price and RSI, volume and price, and volume and RSI.

  - **After Holding Time Variables**:  
    - Gross profit/loss.  
    - Percentage price range between high and low relative to opening price.  
    - Price movement from start to end of the period.

  - **Long-Term Trend Variables**:  
    - Slope of the weekly closing prices over the last 18 periods.  
    - Average weekly closing price over the last 18 periods.

### 4. Target Variable Creation
- **Profitability Calculation**:  
  Define a function to calculate whether a trade was profitable.  
  The target variable `Next_Day_Profitable` is set to 1 if the position was profitable on the next day, 0 otherwise.

### 5. Data Preparation for Modeling
- **Data Cleaning**:  
  Remove unnecessary columns and handle missing values.  
  Ensure consistency in data dimensions.

- **Feature and Target Separation**:  
  Separate features (technical indicators) and the target variable.

- **Data Splitting**:  
  Split the dataset into training and testing sets using an 80-20 split.

- **Data Scaling and Dimensionality Reduction**:  
  Standardize the features using `StandardScaler`.  
  Apply Principal Component Analysis (PCA) to reduce dimensionality while retaining 95% of the variance.

### 6. Model Training and Evaluation
- **Models Used**:  
  - Random Forest Classifier  
  - Logistic Regression  
  - Support Vector Machine (SVM)  
  - Gradient Boosting Classifier  
  - Neural Network (using TensorFlow Keras)

- **Training**:  
  Train each model using the training dataset.

- **Evaluation**:  
  Predict on the test dataset.  
  Evaluate models using:  
  - Accuracy Score  
  - Classification Report (Precision, Recall, F1-Score)  
  - Confusion Matrix

- **Visualization**:  
  - Create bar plots to compare the accuracy of different models.  
  - Generate heatmaps for confusion matrices of each model.

## Results
- **Model Performance**:  
  All models were evaluated, with varying degrees of accuracy.  
  The Neural Network and Random Forest Classifier showed competitive performance.

- **Confusion Matrices**:  
  Provided insights into the models' prediction capabilities, highlighting areas like false positives and false negatives.

- **Visual Comparisons**:  
  Bar plots and heatmaps aided in visualizing and comparing model performances.
