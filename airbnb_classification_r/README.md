# Airbnb Rating Classification Project

This R project performs classification analysis on Airbnb data to predict high vs low ratings (threshold: 3.5).

## Project Structure
- `main.R`: Main analysis script
- `visualizations/`: Directory containing all generated plots
- `data/`: Directory for the dataset (you need to add your dataset here)

## Required R Packages
- tidyverse
- caret
- randomForest
- e1071
- rpart
- rpart.plot
- ggplot2
- corrplot
- scales
- DataExplorer

## How to Run
1. Place your Airbnb dataset in the `data` directory
2. Update the data file name in `main.R`
3. Install required packages using:
   ```R
   install.packages(c("tidyverse", "caret", "randomForest", "e1071", "rpart", "rpart.plot", "ggplot2", "corrplot", "scales", "DataExplorer"))
   ```
4. Run the main.R script

## Features
- Data preprocessing and cleaning
- Exploratory Data Analysis with visualizations
- Implementation of multiple classification models:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - Support Vector Machine (SVM)
  - Naive Bayes
- Model comparison and evaluation
- Visualization of results 