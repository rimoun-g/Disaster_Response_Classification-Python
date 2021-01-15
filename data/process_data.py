import sys
import pandas as pd
import numpy as np
import re
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """This function takes the messages csv files and categories csv file to combine them together as one dataframe"""
    # read messages file
    messages = pd.read_csv(messages_filepath)
    # read categories file
    categories = pd.read_csv(categories_filepath)
    # merge them to make one dataframe
    df = pd.merge(messages,categories)
    return df


def clean_data(df):
    """This function cleans the dataframe and prepares it for applying machine learning"""
    # split the categories into separate columns
    categories = df['categories'].str.split(";", expand=True)
    # save the first row to use them as columns names
    row = categories.head(1).values.tolist()
    # collect the words from the categories names
    category_colnames = list(map(lambda x: x[:-2], row[0]))
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    # drop the old categories column
    df.drop('categories', axis=1, inplace=True)
    # concate the df and the categories dataframes together
    df = pd.concat([df, categories], axis=1)
    # remove the duplicates from the final dataframe
    df.drop_duplicates(inplace=True)
    return df


def save_data(df, database_filename):
    """Save the clean dataframe in a sqlite file"""
    # create sqlite connection
    engine = create_engine('sqlite:///' + database_filename)
    # save the file
    df.to_sql('disaster_clean', engine, index=False, if_exists = 'replace')


def main():
    """This is the main function that runs the file"""
    # get arguments
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]
        
        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        # load data
        df = load_data(messages_filepath, categories_filepath)
        
        # clean data
        print('Cleaning data...')
        df = clean_data(df)
        
        # save the resulted file 
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()