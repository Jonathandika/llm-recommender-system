import dask.dataframe as dd
import time
import os

start_time = time.time()

dtypes = {
    'Id': 'int64',
    'Name': 'object',
    'Authors': 'object',
    'ISBN': 'object',  # ISBN numbers include letters, so use object
    'Rating': 'float64',
    'PublishYear': 'int64',
    'PublishMonth': 'int64',
    'PublishDay': 'int64',
    'Publisher': 'object',
    'RatingDist5': 'object',
    'RatingDist4': 'object',
    'RatingDist3': 'object',
    'RatingDist2': 'object',
    'RatingDist1': 'object',
    'RatingDistTotal': 'object',
    'CountsOfReview': 'int64',
    'Language': 'object',
    'PagesNumber': 'float64',
    'Description': 'object',
    'pagesNumber': 'float64',  # Note: This seems duplicate of 'PagesNumber'
    'Count of text reviews': 'float64'
}

book_dfs = []
for i in os.listdir('data'):
    if i.startswith('book') and i.endswith('csv'):
        df = dd.read_csv('data/' + i, dtype=dtypes, blocksize=4)  # Adjust blocksize if necessary
        book_dfs.append(df)

# Concatenate dataframes
book_df = dd.concat(book_dfs)

book_df = book_df.repartition(npartitions=8)

# Drop duplicates
book_df_cleaned = book_df.drop_duplicates().drop_duplicates(subset=['Id'], keep='last')

# Filter only English books
book_df_eng = book_df_cleaned[(book_df_cleaned['Language']=='eng') | 
                              (book_df_cleaned['Language']=='en-US') | 
                              (book_df_cleaned['Language']=='en-GB')]

# Convert to Parquet format
book_df_eng.to_parquet('data/book_eng_y.parquet', write_index=False)

end_time = time.time()
total_time = end_time - start_time
print(f"Time taken: {total_time} seconds")
