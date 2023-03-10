import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    
    """
    load the two csv files and merge them to one dataframe
    
    input: messages_filepath: the file_path to the messages csv
            categories_filepath: the file path to the categories csv
            
    output: df: the dataframe that merges the two csv files
    
    
    """
    # read the message csv
    messages = pd.read_csv(messages_filepath)
    # read the categories csv
    categories = pd.read_csv(categories_filepath)
    # merge the two dataframes
    df = messages.merge(categories, how = 'outer', on = 'id')
    
    return df

def clean_data (df):
    """
    loads the df dataframe and returns the clean dataframe
    
    input: df: The variable that hold the merged the messages and the categories
    
    output:
    a cleaned df
    """
    categories = df['categories'].str.split(';',expand=True)
    # select the first row of the categories dataframe
    row = categories.iloc[0]
    #extract a list of new column names for categories
    category_colnames = list(map(lambda x: x[:-2],row))
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    for column in categories:
        
    # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
    
    # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    
    # drop the original categories column from `df`
    df.drop('categories', axis = 1, inplace = True)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories], axis = 1)
    # drop duplicate
    df['id'].drop_duplicates(inplace = True)

    df = df[df.related != 2]

   
    return df


def save_data(df, database_filename):
    """
    save the cleaned dataframe in a database
    
    input: df: the clean dataframe
    database_filename: the file path that will hold the dataframe
    
    """
    # define the table name
    table_name = 'disaster_response'
    # define the sql engine
    engine = create_engine('sqlite:///{}'.format(database_filename))
    # save the df to sql
    df.to_sql(table_name, engine, if_exists='replace', index=False,chunksize = 500)  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
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