import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load the spam data
data = pd.read_csv ("C:/Users/Khelendra/Downloads/Final project Spam filter/mail_data1.csv")



# Split the data into features and labels
X = data['Message']
y = data['Category']

# Convert the text data into numerical data using CountVectorizer
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(X)

# Get the length of the email body
body_len = data['Message'].apply(len)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test, body_len_train, body_len_test = train_test_split(X, y, body_len, test_size=0.2, random_state=42)

# Train the spam filter model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predict the labels of the test data
y_pred = model.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Predict spam for new emails
while True:
    new_email = input("Enter a new email: ")
    new_email_len = len(new_email)
    new_email_vector = vectorizer.transform([new_email])
    new_email_spam_prob = model.predict_proba(new_email_vector)[:, 1][0]
    
    if new_email_spam_prob >= 0.5:
        print("This email is spam (spam probability: {:.2f}%)".format(new_email_spam_prob*100))
    else:
        print("This email is not spam (spam probability: {:.2f}%)".format(new_email_spam_prob*100))
        
    print("Email length:", new_email_len)
    print("-----------")
