import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Load and merge messages and categories datasets.
    
    Args:
        messages_filepath (str) : File path of messages dataset
        categories_filepath (str): File path of categories dataset
        
    Returns: 
        df (Dataframe): Merged dataset containing messages and categories.
    """
    # Loading messages dataset
    messages = pd.read_csv(messages_filepath)

    # Loading categories dataset
    categories = pd.read_csv(categories_filepath)

    # Merging datasets on 'id'
    df = messages.merge(categories, on="id")
    
    return df


def clean_data(df):
    """
    Cleans the dataframe by:
    - Splitting the 'categories' column into separate category columns
    - Converting category values to binary (0 or 1)
    - Removing duplicates
    
    Args:
        df (DataFrame): Merged dataframe containing messages and categories.
    
    Returns:
        DataFrame: Cleaned dataframe with separate category columns.
    """
    # Spliting`categories` into separate category columns
    categories = df["categories"].str.split(";", expand=True)

    # Selecting the first row of categories
    row = categories.iloc[0]

    # Extracting category names from the first row
    category_colnames = [category[:-2] for category in row]
    categories.columns = category_colnames

    # Converting category values to numbers (0 or 1)
    for column in categories:
        categories[column] = categories[column].astype(str).str[-1].astype(int)

    # Dropping the original `categories` column from `df`
    df.drop("categories", axis=1, inplace=True)

    # Concatenating `df` with new category columns
    df = pd.concat([df, categories], axis=1)

    # Dropping duplicate rows
    df.drop_duplicates(inplace=True)

    return df


def save_data(df, database_filename):
    """
    Saves the cleaned dataframe into an SQLite database.

    Args:
        df (DataFrame): Cleaned dataframe containing messages and categories.
        database_filename (str): Name of the SQLite database file.
    
    Returns:
        None
    """
    # Create a SQLite database engine
    engine = create_engine(f'sqlite:///{database_filename}')

    # Save dataframe to a SQL table named 'MessagesCategories'
    df.to_sql('MessagesCategories', engine, index=False, if_exists='replace')

    print("✅ Data successfully saved to database!")
    
    


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