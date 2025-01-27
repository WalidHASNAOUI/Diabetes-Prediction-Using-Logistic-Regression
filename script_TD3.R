# data loading 
df <- read.csv("/home/dilaw/Documents/ING-POLYTECH/S8/DecisionalComputerScience/TD03/diabetes(1).csv")

# summarize the data frame 
summary(df)

# Check for missing values 
cat(sum(is.na(df)))

# data visualization 

# univariate data visualization 
hist(
  df$Pregnancies,
  main = "Histrogram of Pregnancies number",
  xlab = "Pregnancies Count",
  breaks = 10
)
hist(
  df$Glucose,
  main = "Histrogram of Glucose concentration",
  xlab = "Glucose concentration",
)
hist(
  df$BloodPressure,
  main = "Histrogram of BloodPressure",
  xlab = "BloodPressure",
)
hist(
  df$SkinThickness,
  main = "Histrogram of SkinThickness",
  xlab = "SkinThickness",
)
hist(
  df$Insulin,
  main = "Histrogram of Insulin",
  xlab = "Insulin",
)
hist(
  df$BMI,
  main = "Histrogram of BMI",
  xlab = "BMI",
)
hist(
  df$DiabetesPedigreeFunction,
  main = "Histrogram of DiabetesPedigreeFunction",
  xlab = "Diabetes  PedigreeFunction",
)
hist(
  df$Age,
  main = "Histrogram of Age",
  xlab = "Age",
)
hist(
  df$Outcome,
  main = "Histrogram of Outcome",
  xlab = "Outcome",
  breaks = 2
)

# Bivariate Visualization 
library(ggplot2)

# Features to plot
features <- c("Glucose", "BloodPressure", "BMI", "Age", "Pregnancies", "DiabetesPedigreeFunction", "Insulin")

# Loop through features to create scatter plots with jitter
for (feature in features) {
  plot <- ggplot(df, aes_string(x = feature, y = "Outcome")) +
    geom_jitter(width = 0.2, height = 0.1, alpha = 0.6, color = "blue") +
    labs(
      title = paste(feature, "vs Outcome"),
      x = feature,
      y = "Outcome (0 or 1)"
    ) +
    theme_minimal()
  
  print(plot)
}

# Part 2 : Split our dataset into : Training(80%) & Test(20%)
library(caTools)

# Set the Random Seed
set.seed(2003)

# Split the Dataset
split <- sample.split(
  df, 
  SplitRatio = 0.8
)

# Create Training and Testing sets
train_df <- subset(
  df, 
  split == TRUE
)
test_df <- subset(
  df, 
  split == FALSE
) 

# check proportions 
prop_train_true <- sum(train_df$Outcome == 1) / length(train_df$Outcome)
prop_train_false <- sum(train_df$Outcome == 0) / length(train_df$Outcome)

prop_test_true <- sum(test_df$Outcome == 1) / length(test_df$Outcome)
prop_test_false <- sum(test_df$Outcome == 0) / length(test_df$Outcome)

prop_original_true <- sum(df$Outcome == 1) / length(df$Outcome)
prop_original_false <- sum(df$Outcome == 0) / length(df$Outcome)


# Compute the Majority Class 
majority_class <- ifelse(
  mean(df$Outcome) > 0.5, 
  1, 
  0
)

# Create a prediction vector where are values are "the majority class"
reference_prediction = rep(
  majority_class, 
  nrow(df)
)

# Compute the accuracy
actual_outcomes <- df$Outcome
reference_accuracy <- mean(
  reference_prediction == actual_outcomes
) 

print(paste("Reference Model Accurancy : ", round(reference_accuracy, 4)*100, "%"))


# Part 3 : Logistic Regression 

# Part 3-1 : Simple Logistic Regression

# Initialize a list where we save fitted models based on each feature (input X)
fitted_models_features <- list()

for (feature in features){
  outcome_model <- glm(
    formula = as.formula(paste("Outcome ~", feature)), 
    data = train_df, 
    family = "binomial"
  )
  
  # summarize the fitted model based on each feature
  print("----------------------------------------------------")
  print(paste("Summarize the fitted model with X is ", feature))
  print(summary(outcome_model)$coefficients)
  print("----------------------------------------------------")
  
  # Save models in a list of models
  fitted_models_features[[feature]] <- outcome_model
}

best_significant_fitted_model <- fitted_models_features$Glucose

# Part 3-2 : Multiple Logistic Regression

# fit the multiple logistic regression
outcome_multiple_model <- glm(
  formula = Outcome ~ Pregnancies + Glucose + BloodPressure + SkinThickness + Insulin + BMI + DiabetesPedigreeFunction + Age,
  data = train_df, 
  family = "binomial"
)

# summarize the multiple logistic regression
summary(outcome_multiple_model)


# Part 4 : Evaluate the model

# Predict results based on "test data" 
pred_proba_y <- predict(
  outcome_multiple_model, 
  test_df, 
  type = "response"
)

# Convert probability to binary class 0 or 1
pred_class_y <- ifelse(
  pred_y > 0.5, 
  1, 
  0
)

# Assign actual values of outcome
true_y <- test_df$Outcome

# Evaluate the prediction - compute the confusion matrix
confusion_matrix <- table(
  true_y, 
  pred_class_y
)

print(confusion_matrix)

# compute the accuracy 
accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)

print(paste("The accurancy of the model is : ", round(accuracy*100, 3), "%"))

