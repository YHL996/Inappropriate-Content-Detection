Project Title: Inappropriate Content Detecting for Craigslist
   
Objectives: This project aimed to build a two-class classification model to predict if posts on Craigslist contain inappropriate content. The model can be used to replace the rule-based filter mechanism so that labor costs can be saved. In addition, the model can be further tuned to 

Methodology: The Python scripts worked on data scraping, labeling, data preprocessing, modeling, model selection, and validation. Our team used Selenium to scrape posts from Craigslist and then utilized a large language model to label data for further model training. For preprocessing, we applied tokenizing, lemmatizing, removal of stopwords, and word embedding. We tried Logistic Regression, Naive Bayes, SVM, Gradient Boosting, and Deep Neural Network for model selection.
   
Results: We selected the Logistic Regression as the final model based on the AUC metric. The Logistic Regression model had the highest score of 0.88.

The presentation file about this project can be found here: https://github.com/YHL996/Inappropriate-Content-Detection/blob/main/Presentation%20file.pdf
