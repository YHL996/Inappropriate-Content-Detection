import csv
import nltk
from nltk.corpus import stopwords
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Function to display confusion matrix
def display_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot(cmap='Blues', values_format='d')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.show()

# CSV file and column information
csv_file_path = 'labeled_data.csv'
column_index = 2  # Assuming the text(title) is in the third column

# Lists to store tokenized reviews and labels
reviews_list = []  # for tokenized & other processing
labels = []

# Initialize lemmatizer
lemmatizer = nltk.stem.WordNetLemmatizer()

# Read reviews and labels from the CSV file
with open(csv_file_path, 'r', encoding='utf-8') as csv_file:
    csv_reader = csv.reader(csv_file)
    for index, row in enumerate(csv_reader):
        if index == 0:
            continue
        if row[column_index] == None or row[column_index+1] == None:
            continue
        if len(row) > column_index:
            #binary label is in 2+3 column 
            label = row[column_index+3]
            review = row[column_index] + " " + row[column_index+1]
            tokens = nltk.word_tokenize(review)
            lemmatized_tokens = [lemmatizer.lemmatize(token).lower() for token in tokens]
            post_processed_tokens = [token for token in lemmatized_tokens if not token in stopwords.words('english') if token.isalpha()]
            reviews_list.append(" ".join(post_processed_tokens))
            labels.append(label)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=20)
vectorizer.fit(reviews_list)
v1 = vectorizer.transform(reviews_list)

# Create a DataFrame with TF-IDF features and labels
feature_names = vectorizer.get_feature_names_out()
tfidf_df = pd.DataFrame(v1.toarray(), columns=feature_names)
tfidf_df['label'] = labels

# Train-Test Split
X = tfidf_df.drop('label', axis=1)
y = tfidf_df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=66)

# Logistic Regression Model
model = LogisticRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
display_confusion_matrix(y_test, predictions, 'Logistic Regression')

# SVM Model
svm_model = SVC(probability=True)
svm_model.fit(X_train, y_train)
svm_predictions = svm_model.predict(X_test)
svm_accuracy = accuracy_score(y_test, svm_predictions)
# display_confusion_matrix(y_test, svm_predictions, 'SVM')

# Naive Bayes Model
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
nb_predictions = nb_model.predict(X_test)
nb_accuracy = accuracy_score(y_test, nb_predictions)
# display_confusion_matrix(y_test, nb_predictions, 'Naive Bayes')

# Gradient Boosting Model
gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.15, max_depth=3)
gb_model.fit(X_train, y_train)
gb_predictions = gb_model.predict(X_test)
gb_accuracy = accuracy_score(y_test, gb_predictions)
# display_confusion_matrix(y_test, gb_predictions, 'Gradient Boosting')

# Neural Network Model
DLmodel = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(5, 3, 3, 3                                                                                                                     , 3), activation='relu', random_state=1, max_iter=2000)
DLmodel.fit(X_train, y_train)
y_pred_DL = DLmodel.predict(X_test)
acc_DL = accuracy_score(y_test, y_pred_DL)
# display_confusion_matrix(y_test, y_pred_DL, 'Deep Learning')

# Assuming 'new_sample' is the new text you want to predict
content1 = "Are you seriously here browsing these posts, expecting to meet someone to quarantine and chill with?! The only stuff you will certainly see allow me to share bunk advertisements as well as the coppers trying to arrest dudes looking to get easy ⓢEⓧ Prevent wasting your time and risking your safety when we curently have thus much to worry about together with the outbreak! Let me give you the last place left filled with everyday babes that are serious about sharing photographs and meeting to chill. Just ⒼOOⒼLⒺ the name 🄱🄰🄽🄶🄵🄾🅁🄵🅄🄽. 🄲🄾🄼 and you can view it right in the top While you are ready to stop wasting time, move see for your own. You will be cheerful you did!"
content2 = "'26 years old sexxy Girl,,,alone,,,,, ' You must be one batshit crazy dude if you honestly think real women come here to find someone to chill with?! The days of genuine posts and engaging conversations are long gone, replaced by a sea of BS and trash Don't risk your job, relationship, or freedom replying to these posts! There's still one sight that hasn't been tainted by cheaters or the feds. It's called ⓅⓊⓈⓈⓎⒷⒾⓏ and it's a haven for everyday girls who are just looking for some company. All you have to do is search for the name or type it in your browser. It's like finding a pot of gold at the end of a rainbow! Go see for yourself before its ruined like the rest. Have fun and be well!"
content3 = "Older Women (Silver Spring) Mid 40’s SBM loves the company of women 50+. Extra points if unshaven"
content4 = "Pen pal (Akron) Older guy looking for someone to chat with. In my mid 60s and would prefer someone around my age. And preferably a female. Easy to talk to, non judgmental. Someone?" 

# Assuming 'new_samples' is a list of new texts you want to predict
new_samples = [content1, content2, content3, content4]

# New samples preprocessing
processed_samples = []
for sample in new_samples:
    tokens = nltk.word_tokenize(sample)
    lemmatized_tokens = [lemmatizer.lemmatize(token).lower() for token in tokens]
    post_processed_tokens = [token for token in lemmatized_tokens if not token in stopwords.words('english') if token.isalpha]
    processed_text = " ".join(post_processed_tokens)
    processed_samples.append(processed_text)

# Predictions for new samples
logistic_predictions = model.predict(vectorizer.transform(new_samples))
svm_predictions_new_samples = svm_model.predict(vectorizer.transform(new_samples).toarray())
nb_predictions_new_samples = nb_model.predict(vectorizer.transform(new_samples))
gb_predictions_new_samples = gb_model.predict(vectorizer.transform(new_samples))
dl_predictions = DLmodel.predict(vectorizer.transform(new_samples))

# Display or use the predictions
print("Logistic Regression Predictions for new samples:", logistic_predictions)
print("SVM Predictions for new samples:", svm_predictions_new_samples)
print("Naive Bayes Predictions for new samples:", nb_predictions_new_samples)
print("Gradient Boosting Predictions for new samples:", gb_predictions_new_samples)
print("Deep Learning Predictions for new samples:", dl_predictions)

# Model Evaluation
# Logistic Regression
logistic_probabilities = model.predict_proba(X_test)[:, 1]
logistic_auc = roc_auc_score(y_test.astype(int), logistic_probabilities)
print(f"Logistic Regression AUC: {logistic_auc}")

# SVM
svm_probabilities = svm_model.predict_proba(X_test)[:, 1]
svm_auc = roc_auc_score(y_test.astype(int), svm_probabilities)
print(f"SVM AUC: {svm_auc}")

# Naive Bayes
nb_probabilities = nb_model.predict_proba(X_test)[:, 1]
nb_auc = roc_auc_score(y_test.astype(int), nb_probabilities)
print(f"Naive Bayes AUC: {nb_auc}")

# Gradient Boosting
gb_probabilities = gb_model.predict_proba(X_test)[:, 1]
gb_auc = roc_auc_score(y_test.astype(int), gb_probabilities)
print(f"Gradient Boosting AUC: {gb_auc}")

# Deep Learning
dl_probabilities = DLmodel.predict_proba(X_test)[:, 1]
dl_auc = roc_auc_score(y_test.astype(int), dl_probabilities)
print(f"Deep Learning AUC: {dl_auc}")

# Plot ROC curves for each model
fpr_logistic, tpr_logistic, _ = roc_curve(y_test.astype(int), logistic_probabilities)
fpr_svm, tpr_svm, _ = roc_curve(y_test.astype(int), svm_probabilities)
fpr_nb, tpr_nb, _ = roc_curve(y_test.astype(int), nb_probabilities)
fpr_gb, tpr_gb, _ = roc_curve(y_test.astype(int), gb_probabilities)
fpr_dl, tpr_dl, _ = roc_curve(y_test.astype(int), dl_probabilities)

plt.figure(figsize=(10, 8))
plt.plot(fpr_logistic, tpr_logistic, label=f'Logistic Regression (AUC = {logistic_auc:.2f})')
plt.plot(fpr_svm, tpr_svm, label=f'SVM (AUC = {svm_auc:.2f})')
plt.plot(fpr_nb, tpr_nb, label=f'Naive Bayes (AUC = {nb_auc:.2f})')
plt.plot(fpr_gb, tpr_gb, label=f'Gradient Boosting (AUC = {gb_auc:.2f})')
plt.plot(fpr_dl, tpr_dl, label=f'Deep Learning (AUC = {dl_auc:.2f})')

plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

