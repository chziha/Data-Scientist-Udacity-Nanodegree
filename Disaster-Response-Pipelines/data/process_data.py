import sys
import os
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    Function to return a dataframe by merging two imported CSV files
    with duplicate rows removed

    Args:
        messages_filepath: a string for the path of messages CSV file
        categories_filepath: a string for the path of categories CSV file

    Returns:
        df: a Dataframe containg the merged data from two inputs
    '''
    
    try:
        messages = pd.read_csv(messages_filepath)
        categories = pd.read_csv(categories_filepath)
        # Remove duplicate rows from both dataframes
        messages.drop_duplicates(inplace=True)
        categories.drop_duplicates(inplace=True)
        df = messages.merge(categories, how='inner', on='id')
        return df
    except:
        print('Fail to open the specified CSV files.')


def clean_data(df):
    '''
    Function to clean the data to prepare for the ML pipeline

    Agrs:
        df: a Dataframe obtained from the load_data function

    Returns:
        df: a cleaned Dataframe prepared for the ML pipeline
    '''

    # Create a dataframe of the 36 individual category columns
    cats_ind = df.categories.str.split(';', expand=True)

    # Extract a list of new column names for categories
    row = cats_ind.iloc[0, :]
    category_colnames = [name.split('-')[0] for name in row]
    cats_ind.columns = category_colnames

    # Set each value to be the last character of the string and convert to int
    for column in cats_ind:
        cats_ind[column] = cats_ind[column].str[-1]
        cats_ind[column] = cats_ind[column].astype('int')

    # Concatenate two dataframes with the original categories column dropped
    df.drop(['categories'], axis=1, inplace = True)
    df = pd.concat([df, cats_ind], axis=1)
    df.drop_duplicates(inplace=True)

    # Change all values of 2 to 1 in related category
    df['related'] = df['related'].apply(lambda x: 1 if x == 2 else x)
    return df



def save_data(df, database_filepath):
    '''
    Function to save the cleaned data to a database

    Args:
        df: a Dataframe prepared for the ML pipeline
        database_filepath: a string for the path of the database file

    Returns:
        None
    '''

    engine = create_engine('sqlite:///{}'.format(database_filepath))

    # Extract the table name from the database file path
    tab_name = database_filepath.split('/')[-1].split('.')[0]
    df.to_sql(tab_name, engine, if_exists='replace', index=False)  


def main():
    '''
    Function to run the ETL pipeline
    '''

    # Check if the correct number of parameters is passed
    if len(sys.argv) == 4:
        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        print('Data cleaned!')
        
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