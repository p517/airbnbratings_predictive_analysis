#install.packages(c("tidyverse", "caret", "randomForest", "e1071", "rpart", "rpart.plot", "ggplot2", "corrplot", "scales", "DataExplorer"))# Required Libraries
library(tidyverse)
library(caret)
library(randomForest)
library(e1071)
library(rpart)
library(rpart.plot)
library(ggplot2)
library(corrplot)
library(scales)
library(DataExplorer)
library(pROC)  # For ROC curves

# Set seed for reproducibility
set.seed(123)

# Data Loading and Preprocessing
data <- read.csv("D:/Bhoomi/UTD/Academic/Sem-2/MIS - 6356 BA with R/Project/airbnb_classification_r/data/airbnb-2.csv")

# Data Preprocessing Function
preprocess_data <- function(data) {
  # Convert rating to numeric (removing 'New' and converting to numeric)
  data$rating_numeric <- as.numeric(gsub("New", NA, as.character(data$rating)))
  
  # Create binary classification using 4.9 as threshold
  data$high_rating <- factor(ifelse(data$rating_numeric >= 4.9, "high", "low"),
                            levels = c("low", "high"))
  
  # Convert reviews to numeric
  data$reviews <- as.numeric(as.character(data$reviews))
  
  # Select and prepare features for modeling
  selected_features <- c("reviews", "price", "bathrooms", "beds", "guests", 
                        "bedrooms", "high_rating")
  
  data <- data[, selected_features]
  
  # Handle missing values
  data <- na.omit(data)
  
  # Log transform price due to its skewed distribution
  data$price <- log1p(data$price)
  
  return(data)
}

# Exploratory Data Analysis
perform_eda <- function(data) {
  # Create directory for visualizations
  dir.create("visualizations", showWarnings = FALSE)
  
  # Function to remove outliers for visualization
  remove_outliers <- function(x) {
    qnt <- quantile(x, probs=c(.05, .95), na.rm = TRUE)
    H <- 1.5 * IQR(x, na.rm = TRUE)
    x[x < (qnt[1] - H)] <- qnt[1]
    x[x > (qnt[2] + H)] <- qnt[2]
    return(x)
  }
  
  # Create visualization data without outliers
  viz_data <- data
  viz_data$price_clean <- remove_outliers(exp(data$price)-1)
  viz_data$reviews_clean <- remove_outliers(data$reviews)
  
  # 1. Class Distribution with Percentages
  total_count <- nrow(data)
  dist_data <- as.data.frame(table(data$high_rating))
  dist_data$percentage <- dist_data$Freq / total_count * 100
  
  p1 <- ggplot(dist_data, aes(x = Var1, y = percentage, fill = Var1)) +
    geom_bar(stat = "identity", width = 0.6) +
    geom_text(aes(label = sprintf("%.1f%%", percentage)), 
              position = position_stack(vjust = 0.5)) +
    labs(title = "Distribution of Airbnb Ratings",
         subtitle = "Threshold: 4.9 stars",
         x = "Rating Category",
         y = "Percentage") +
    scale_fill_manual(values = c("#FF9999", "#99FF99"),
                     labels = c("Good (< 4.9)", "Exceptional (≥ 4.9)"),
                     name = "Rating Category") +
    theme_minimal() +
    theme(legend.position = "bottom")
  ggsave("visualizations/1_rating_distribution.png", p1, width = 10, height = 6)

  # 2. Price vs Rating with Density
  p2 <- ggplot(viz_data, aes(x = price_clean, fill = high_rating)) +
    geom_density(alpha = 0.7) +
    scale_x_continuous(labels = scales::dollar_format(), 
                      breaks = seq(0, max(viz_data$price_clean), by = 2000)) +
    labs(title = "Price Distribution by Rating Category",
         subtitle = "Higher-rated properties tend to be more expensive",
         x = "Price (USD)",
         y = "Density") +
    scale_fill_manual(values = c("#FF9999", "#99FF99"),
                     labels = c("Good (< 4.9)", "Exceptional (≥ 4.9)"),
                     name = "Rating Category") +
    theme_minimal() +
    theme(legend.position = "bottom",
          axis.text.x = element_text(angle = 45, hjust = 1))
  ggsave("visualizations/2_price_distribution.png", p2, width = 12, height = 6)

  # 3. Feature Importance Plot
  numeric_cols <- c("reviews", "price", "bathrooms", "beds", "guests", "bedrooms")
  importance_data <- data.frame(
    Feature = numeric_cols,
    Correlation = sapply(numeric_cols, function(col) {
      cor(as.numeric(data$high_rating == "high"), data[[col]], 
          use = "complete.obs")
    })
  )
  importance_data$Feature <- factor(importance_data$Feature, 
                                  levels = importance_data$Feature[order(abs(importance_data$Correlation))])
  
  p3 <- ggplot(importance_data, aes(x = Feature, y = abs(Correlation))) +
    geom_bar(stat = "identity", fill = "steelblue") +
    geom_text(aes(label = sprintf("%.3f", Correlation)), vjust = -0.5) +
    labs(title = "Feature Importance for High Ratings",
         subtitle = "Absolute correlation with rating category",
         x = "Feature",
         y = "Absolute Correlation") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
  ggsave("visualizations/3_feature_importance.png", p3, width = 10, height = 6)

  # 4. Reviews vs Price by Rating Category
  p4 <- ggplot(viz_data, aes(x = price_clean, y = reviews_clean, color = high_rating)) +
    geom_point(alpha = 0.5) +
    geom_smooth(method = "loess", se = FALSE) +
    scale_x_continuous(labels = scales::dollar_format()) +
    labs(title = "Relationship between Price, Reviews, and Ratings",
         subtitle = "Higher-priced properties tend to have fewer reviews",
         x = "Price (USD)",
         y = "Number of Reviews") +
    scale_color_manual(values = c("#FF9999", "#99FF99"),
                      labels = c("Good (< 4.9)", "Exceptional (≥ 4.9)"),
                      name = "Rating Category") +
    theme_minimal() +
    theme(legend.position = "bottom")
  ggsave("visualizations/4_price_reviews_relationship.png", p4, width = 12, height = 6)

  # 5. Summary Statistics Table
  stats_table <- data.frame(
    Metric = c("Average Price", "Median Price", 
               "Average Reviews", "Median Reviews"),
    Low_Rating = c(
      mean(viz_data$price_clean[viz_data$high_rating == "low"]),
      median(viz_data$price_clean[viz_data$high_rating == "low"]),
      mean(viz_data$reviews_clean[viz_data$high_rating == "low"]),
      median(viz_data$reviews_clean[viz_data$high_rating == "low"])
    ),
    High_Rating = c(
      mean(viz_data$price_clean[viz_data$high_rating == "high"]),
      median(viz_data$price_clean[viz_data$high_rating == "high"]),
      mean(viz_data$reviews_clean[viz_data$high_rating == "high"]),
      median(viz_data$reviews_clean[viz_data$high_rating == "high"])
    )
  )
  
  write.csv(stats_table, "visualizations/5_summary_statistics.csv", row.names = FALSE)
  
  # Print summary of findings
  cat("\nKey Insights from the Analysis:\n")
  cat("1. Rating Distribution:", 
      sprintf("%.1f%% Low (< 4.9), %.1f%% High (≥ 4.9)\n", 
              dist_data$percentage[1], dist_data$percentage[2]))
  cat("2. Price Difference:", 
      sprintf("High-rated properties are $%.2f more expensive on average\n",
              stats_table$High_Rating[1] - stats_table$Low_Rating[1]))
  cat("3. Review Patterns:", 
      sprintf("High-rated properties have %.1f fewer reviews on average\n",
              stats_table$Low_Rating[3] - stats_table$High_Rating[3]))
}

# Model Training and Evaluation
train_evaluate_models <- function(data) {
  # Split data into training and testing sets
  set.seed(123)
  train_index <- createDataPartition(data$high_rating, p = 0.8, list = FALSE)
  train_data <- data[train_index, ]
  test_data <- data[-train_index, ]
  
  # List to store model results
  model_results <- list()
  
  # Function to calculate metrics
  calculate_metrics <- function(predictions, actuals, probabilities = NULL) {
    cm <- confusionMatrix(predictions, actuals)
    metrics <- list(
      accuracy = cm$overall["Accuracy"],
      precision = cm$byClass["Precision"],
      recall = cm$byClass["Recall"],
      f1_score = cm$byClass["F1"],
      specificity = cm$byClass["Specificity"]
    )
    
    if (!is.null(probabilities)) {
      roc_obj <- roc(actuals, probabilities)
      metrics$auc <- auc(roc_obj)
      metrics$roc <- roc_obj
    }
    
    return(metrics)
  }
  
  # Control parameters for cross-validation
  ctrl <- trainControl(method = "cv", 
                      number = 5,
                      classProbs = TRUE,
                      summaryFunction = twoClassSummary)
  
  # 1. Logistic Regression
  print("Training Logistic Regression...")
  log_model <- train(high_rating ~ .,
                    data = train_data,
                    method = "glm",
                    family = "binomial",
                    trControl = ctrl,
                    metric = "ROC")
  log_pred <- predict(log_model, test_data)
  log_prob <- predict(log_model, test_data, type = "prob")[,"high"]
  model_results$logistic <- calculate_metrics(log_pred, test_data$high_rating, log_prob)
  
  # 2. Decision Tree
  print("Training Decision Tree...")
  dt_model <- rpart(high_rating ~ .,
                   data = train_data,
                   method = "class",
                   control = rpart.control(cp = 0.01))
  dt_pred <- predict(dt_model, test_data, type = "class")
  dt_prob <- predict(dt_model, test_data, type = "prob")[,"high"]
  model_results$decision_tree <- calculate_metrics(dt_pred, test_data$high_rating, dt_prob)
  
  # Save decision tree visualization
  png("visualizations/decision_tree.png", width = 1200, height = 800)
  rpart.plot(dt_model, main = "Decision Tree for Rating Classification",
             extra = 101,
             box.palette = "RdYlGn",
             shadow.col = "gray")
  dev.off()
  
  # 3. Random Forest
  print("Training Random Forest...")
  rf_model <- randomForest(high_rating ~ .,
                          data = train_data,
                          importance = TRUE,
                          ntree = 500)
  rf_pred <- predict(rf_model, test_data)
  rf_prob <- predict(rf_model, test_data, type = "prob")[,"high"]
  model_results$random_forest <- calculate_metrics(rf_pred, test_data$high_rating, rf_prob)
  
  # Feature importance plot for Random Forest
  importance_df <- as.data.frame(importance(rf_model))
  importance_df$Feature <- rownames(importance_df)
  p3 <- ggplot(importance_df, aes(x = reorder(Feature, MeanDecreaseGini), y = MeanDecreaseGini)) +
    geom_bar(stat = "identity", fill = "steelblue") +
    coord_flip() +
    labs(title = "Feature Importance (Random Forest)",
         x = "Features",
         y = "Mean Decrease in Gini") +
    theme_minimal()
  ggsave("visualizations/feature_importance.png", p3, width = 10, height = 6)
  
  # 4. SVM
  print("Training SVM...")
  svm_model <- svm(high_rating ~ .,
                   data = train_data,
                   kernel = "radial",
                   probability = TRUE)
  svm_pred <- predict(svm_model, test_data)
  svm_prob <- attr(predict(svm_model, test_data, probability = TRUE), "probabilities")[,"high"]
  model_results$svm <- calculate_metrics(svm_pred, test_data$high_rating, svm_prob)
  
  # 5. Naive Bayes
  print("Training Naive Bayes...")
  nb_model <- naiveBayes(high_rating ~ .,
                        data = train_data)
  nb_pred <- predict(nb_model, test_data)
  nb_prob <- predict(nb_model, test_data, type = "raw")[,"high"]
  model_results$naive_bayes <- calculate_metrics(nb_pred, test_data$high_rating, nb_prob)
  
  # Plot ROC curves
  png("visualizations/roc_curves.png", width = 1000, height = 800)
  plot(model_results$logistic$roc, col = "blue", main = "ROC Curves Comparison")
  plot(model_results$decision_tree$roc, col = "red", add = TRUE)
  plot(model_results$random_forest$roc, col = "green", add = TRUE)
  plot(model_results$svm$roc, col = "purple", add = TRUE)
  plot(model_results$naive_bayes$roc, col = "orange", add = TRUE)
  legend("bottomright", 
         legend = c(paste("Logistic (AUC =", round(model_results$logistic$auc, 3), ")"),
                   paste("Decision Tree (AUC =", round(model_results$decision_tree$auc, 3), ")"),
                   paste("Random Forest (AUC =", round(model_results$random_forest$auc, 3), ")"),
                   paste("SVM (AUC =", round(model_results$svm$auc, 3), ")"),
                   paste("Naive Bayes (AUC =", round(model_results$naive_bayes$auc, 3), ")")),
         col = c("blue", "red", "green", "purple", "orange"),
         lwd = 2)
  dev.off()
  
  return(model_results)
}

# Compare Models
compare_models <- function(model_results) {
  # Create comparison dataframe
  metrics <- c("Accuracy", "Precision", "Recall", "F1 Score", "Specificity", "AUC")
  models <- names(model_results)
  
  results_df <- data.frame(
    Model = models,
    Accuracy = sapply(model_results, function(x) x$accuracy),
    Precision = sapply(model_results, function(x) x$precision),
    Recall = sapply(model_results, function(x) x$recall),
    F1_Score = sapply(model_results, function(x) x$f1_score),
    Specificity = sapply(model_results, function(x) x$specificity),
    AUC = sapply(model_results, function(x) x$auc)
  )
  
  # Plot comparison
  results_long <- gather(results_df, key = "Metric", value = "Value", -Model)
  p4 <- ggplot(results_long, aes(x = Model, y = Value, fill = Metric)) +
    geom_bar(stat = "identity", position = "dodge") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
    labs(title = "Model Comparison",
         y = "Score",
         x = "Model") +
    scale_fill_brewer(palette = "Set3")
  ggsave("visualizations/model_comparison.png", p4, width = 12, height = 6)
  
  # Save results to CSV
  write.csv(results_df, "visualizations/model_comparison_results.csv", row.names = FALSE)
  
  return(results_df)
}

# Main execution
main <- function() {
  # Create directories if they don't exist
  dir.create("visualizations", showWarnings = FALSE)
  
  # Load and preprocess data
  print("Loading and preprocessing data...")
  data <- preprocess_data(data)
  
  # Perform EDA
  print("Performing Exploratory Data Analysis...")
  perform_eda(data)
  
  # Train and evaluate models
  print("Training and evaluating models...")
  model_results <- train_evaluate_models(data)
  
  # Compare models
  print("Comparing models...")
  comparison_results <- compare_models(model_results)
  print("\nModel Comparison Results:")
  print(comparison_results)
  
  print("\nAnalysis complete! Check the 'visualizations' directory for results.")
}

# Run the analysis
main() 