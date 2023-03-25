from bs4 import BeautifulSoup
import requests
import pandas as pd
import csv

def process_urls(df):
    for index, row in df.iterrows():
        # Initialize the object with the document
        doc = row.iloc[1]  # Access the second column
        # getting response object
        res = requests.get(doc)
        # Initialize the object with the document
        soup = BeautifulSoup(res.content, "html.parser")
        # Get the whole body tag
        tag = soup.body
        # Extract strings from body tag and join them
        content = ' '.join(string for string in tag.strings)
        # Write the content to the third column of df
        df.at[index, 'content'] = content
    return df

path_to_csv = 'gunthy_wiki_links.csv'
df = pd.read_csv(path_to_csv)

if __name__ == "__main__":
    print(df.columns)  # Print the columns before calling process_urls
    df_processed = process_urls(df)
    df_processed.to_csv('gunthy_wiki_model_001.csv', index=False)


