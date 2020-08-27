# import libraries
import sys
import pandas as pd
import re
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Load and merage Message and categories datset.
    """
    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, how='inner', on='id')
    return df

def clean_data(df):
    """
    Split categories into separate category columns and convert category values to 0 or 1.
    
    """
    
    # split categories into separate category columns
    categories = df['categories'].str.split(';', expand=True)
    
    # rename the columns of `categories`
    row = categories.iloc[0,:]
    category_colnames = row.str.split('-', expand=True).iloc[:,0].values
    categories.columns = category_colnames
    
    # set each value to be the last character of the string
    for column in categories:      
        categories[column] = categories[column].str.replace(r"\w+-","")    
        categories[column] = categories[column].astype('Int64')
      
    # drop 'child_alone' column because only contain one value 0
    categories.drop(['child_alone'], axis=1, inplace= True)    
    #drop columns which are not meaningful categories for disaster type
    categories.drop(['related', 'request', 'offer', 'direct_report'], axis=1, inplace= True)     
    # get new categories names
    category_colnames = categories.columns
    
    
    # replace categories column in df with new category columns
    df.drop(['categories'], axis=1, inplace=True)
    df = pd.concat([df,categories], axis=1)
    
    # drop duplicated rows
    df.drop(df[df.duplicated(keep='first')].index, axis=0, inplace=True)
    # drop row do not been categorized 
    df = df.loc[~(df[category_colnames]==0).all(axis=1)]
    print(df.shape)
    return df
    

def save_data(df, database_filename):
    """
    Save cleaned data to database_filename
    """
    
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('DisasterResponse', engine, index=False, if_exists='replace')  


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
