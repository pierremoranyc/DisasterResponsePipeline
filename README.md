Project Overview

In the event of disasters, thousands of messages are received by emergency response teams via social media and disaster relief organizations. These messages need to be categorized quickly to prioritize and route them efficiently.

This project automates the classification of messages using Natural Language Processing (NLP) and Machine Learning.
	•	The ETL pipeline extracts data from a database, transforms and cleans it.
	•	The Machine Learning model trains on labeled data to classify messages.
	•	The Flask Web App allows users to enter new messages and get real-time classifications.
    
    
Dataset

The dataset comes from Figure Eight, containing real-world disaster messages. The dataset consists of:
	•	messages.csv - Raw disaster messages
	•	categories.csv - Classification labels for each message

Each message is labeled into 36 different categories (e.g., medical help, water shortage, fire, aid-related, infrastructure damage).



Installation

To run this project, install the required dependencies:
pip install -r requirements.txt

ETL Pipeline

The ETL pipeline (process_data.py) performs the following steps:
	1.	Load Data: Reads disaster_messages.csv and disaster_categories.csv.
	2.	Clean Data: Splits categories into separate columns, converts to numeric format.
	3.	Store Data: Saves cleaned data to an SQLite database (DisasterResponse.db).

To run the ETL pipeline:
python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db

Machine Learning Pipeline

The train_classifier.py script:
	1.	Loads Cleaned Data from DisasterResponse.db
	2.	Tokenizes Messages using NLTK
	3.	Trains a Machine Learning Model using RandomForestClassifier
	4.	Optimizes Hyperparameters using GridSearchCV
	5.	Saves Model as classifier.pkl (Dynamically downloaded from Google Drive)

To train the model:
python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl

Web Application

The Flask web app allows users to input messages and see real-time classifications.

To run the app:
python app/run.py
Then, open http://0.0.0.0:3000/ in your browser.
