import pandas as pd
import numpy as np
from dotenv import dotenv_values
import dask.dataframe as dd
from dask.multiprocessing import get
from FlagEmbedding import FlagModel
from helper.PredictRating import PredictRating
from functools import wraps
import time
import csv

current_time = time.strftime("%Y%m%d-%H%M%S")
file_handler = open(f'output/function_timings_nonparallel_{current_time}.txt', 'w')

def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        file_handler.write(f'Function {func.__name__}--{kwargs["name"] if len(kwargs)>0 else ""} Took {total_time:.4f} seconds\n')
        
        return result
    return timeit_wrapper


class RecommendationSystem():
    def __init__(self):
        self.model = FlagModel('BAAI/bge-large-en-v1.5', 
                        query_instruction_for_retrieval="Represent this sentence for searching relevant passages: ",
                        use_fp16=False)

    def embed_text(self, doc):
        return self.model.encode(doc)

    @timeit
    def load_data(self):
        book_df = pd.read_parquet('data/book_eng.parquet')
        book_df_cleaned = book_df.dropna(subset = ['Description'])
        book_df_cleaned.reset_index(drop = True, inplace = True)
        sample_book_cleaned = book_df_cleaned.sample(1000, random_state=42) #delete

        user_rating_df = pd.read_parquet('data/user_rating_total.parquet')
        user_rating_df_cleaned = user_rating_df.drop_duplicates()
        user_rating_df_cleaned.dropna(inplace=True)
        user_rating_df_cleaned.reset_index(drop = True, inplace = True)
        
        return sample_book_cleaned, user_rating_df_cleaned

    @timeit
    def generate_embeddings(self, df, **kwargs):
        embeddings_desc = df['Description'].apply(self.embed_text)
        embeddings_title = df['Name'].apply(self.embed_text)

        embeddings_desc_df = pd.DataFrame(embeddings_desc.tolist())
        embeddings_title_df = pd.DataFrame(embeddings_title.tolist())

        for embeddings_df in [embeddings_desc_df, embeddings_title_df]:
            embeddings_df.columns = [str(x) for x in range(embeddings_df.shape[1])]
            embeddings_df.insert(0, 'Id', df['Id'].tolist())
        
        return embeddings_desc_df, embeddings_title_df

    @timeit
    def compute_similarity(self, embeddings, **kwargs):
        embeddings = embeddings.iloc[:, 1:]
        similarity_matrix = np.matrix(np.zeros((embeddings.shape[0], embeddings.shape[0],)))
        for i in range(0,embeddings.shape[0]):
            for j in range(0,embeddings.shape[0]):
                similarity_matrix[i, j] = np.sum(embeddings.iloc[i, :] * embeddings.iloc[j, :])

        return pd.DataFrame(similarity_matrix)

    @timeit
    def generate_user_item_matrix(self, final_similarity_matrix, user_rating_df):
        existing_books = [str(x) for x in final_similarity_matrix.columns]
        user_rating_df['book_id'] = user_rating_df['book_id'].astype('int').astype('str')
        user_rating_with_existing_books = user_rating_df[user_rating_df['book_id'].isin(existing_books)]
        user_rating_with_existing_books.ID = user_rating_with_existing_books.ID.astype('str')

        df_aggregated = user_rating_with_existing_books.groupby(['ID', 'book_id'])[['Rating']].mean().reset_index()
        user_item_matrix = df_aggregated.pivot(index='ID', columns='book_id', values='Rating').fillna(0)

        return user_item_matrix

    @timeit
    def get_top_k_recommendations(self, k, filled_matrix, book_df):
        print(f'======= Getting Top {k} Recommendations =======')
        user_id_list = []
        book_metadata_title_list = []
        book_metadata_description_list = []
        book_metadata_id_list = []
        for user_id in filled_matrix.index:
            book_metadata_id = filled_matrix.T[user_id].sort_values(ascending = False)[:10].index.tolist()
            book_metadata_id_list.extend(book_metadata_id)
            user_id_list.extend([user_id]*len(book_metadata_id))
            for book_id in book_metadata_id:
                book_metadata_title_list.append(book_df[book_df['Id']==int(book_id)]['Name'].values[0])
                book_metadata_description_list.append(book_df[book_df['Id']==int(book_id)]['Description'].values[0])
        
        return pd.DataFrame({
            'user_id': user_id_list, 
            'book_id': book_metadata_id_list, 
            'book_title': book_metadata_title_list, 
            'description': book_metadata_description_list
            })

    @timeit
    def generate_recommendations(self):
        #1. Load data
        print('======= Loading Data =======')
        book_df, user_rating_df = self.load_data()

        #2. Generate embeddings -- Description
        print('======= Generating Embeddings =======')
        embeddings_desc_df, embeddings_title_df = self.generate_embeddings(book_df)

        #4. Item similarity -- item x item matrix
        # Description
        print('======= Generating item similarity -- Desc =======')
        similarity_matrix_desc = self.compute_similarity(embeddings_desc_df, name="description")

        # Title
        print('======= Generating item similarity -- Title =======')
        similarity_matrix_title = self.compute_similarity(embeddings_title_df, name='title')

        # Final Item Similarity Matrix
        print('======= Generating final item similarity =======')
        final_similarity_matrix = 0.4*similarity_matrix_title + 0.6*similarity_matrix_desc
        final_similarity_matrix.columns = embeddings_title_df['Id'].tolist()
        final_similarity_matrix.index = embeddings_title_df['Id'].tolist()
        final_similarity_matrix.columns = final_similarity_matrix.columns.astype('str')
        final_similarity_matrix.index = final_similarity_matrix.index.astype('str')

        #5. Generate User Item Matrix (Ratings)
        print('======= Generating user-item matrix =======')
        user_item_matrix = self.generate_user_item_matrix(final_similarity_matrix, user_rating_df)
        
        #6. Generate Recommendation Table -- Predict rating of unrated books
        print('======= Predicting =======')

        pr = PredictRating()
        filled_matrix = pr.fill_user_item_matrix_nonparallel(user_item_matrix, final_similarity_matrix)

        #7. Generate Top K Recommendations
        recommendations = self.get_top_k_recommendations(10, filled_matrix, book_df)

        recommendations.to_parquet(f'output/top_k_recommendations_nonparallel_{current_time}.parquet')

        print('======= Done =======')
        return

if __name__ == '__main__':
    config = dotenv_values(".env")
    rs = RecommendationSystem()
    rs.generate_recommendations()

file_handler.close()