# Disaster Response Pipeline Project

### Project idea: 
Classify text messages based on categories.


### Description of project files
--- data (folder)
	--- disaster_categories.csv (categories data)
    --- disaster_messages.csv (messages data)
    --- DisasterResponse.db (database to save clean data)
    --- process_data.py (processes all the data in stores the cleaned data in sqlite file)

--- models (folder)
	--- train_classifier (trains the classification model and saves the model as a pickle file)
	--- classifier.pkl (the saved model)
    
--- app (folder)
	--- run.py (run the web app in debug mode)
    --- templates (folder)
    	--- go.html (extends to the master page functionalities)
        --- master.html (The home page of the application)

---- README.md

### Project Summary:
a lot of people around the word send messages related to disasters and aid requests. Using previous messages can be helpful to predict the topic and the aspects of the new messages to direct them to the proper channels to get the aid or help that they request or to receive the information from others.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
