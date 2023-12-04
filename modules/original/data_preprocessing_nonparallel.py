import os
import pandas as pd
import time

start_time = time.time()

book_dfs = []

for i in os.listdir('data'):
    if i.startswith('book') and i.endswith('csv'):
        book_dfs.append(pd.read_csv('data/' + i))

book_df = pd.concat(book_dfs)

book_df_cleaned = book_df.drop_duplicates()
book_df_cleaned.reset_index(drop = True, inplace = True)
book_df_cleaned = book_df_cleaned.drop_duplicates(subset=['Id'], keep='last')
book_df_cleaned.reset_index(drop = True, inplace = True)

# Filter only english books
book_df_eng = book_df_cleaned[(book_df_cleaned['Language']=='eng') | (book_df_cleaned['Language']=='en-US') | (book_df_cleaned['Language']=='en-GB')]

book_df_eng.to_parquet('data/book_eng_x.parquet')

end_time = time.time()
total_time = end_time - start_time
print(f"Time taken: {total_time} seconds")