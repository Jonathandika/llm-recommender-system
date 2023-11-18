import pandas as pd
import numpy as np
from dotenv import dotenv_values
import dask.dataframe as dd
from dask.multiprocessing import get
from FlagEmbedding import FlagModel
from modules.helper.PredictRating import PredictRating
from functools import wraps
import time
import csv
import pinecone
from langchain.embeddings import HuggingFaceEmbeddings
from tqdm import tqdm
import concurrent
import concurrent.futures

file_handler = open('data/function_timings_parallelized.txt', 'w')

def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        file_handler.write(f'Function {func.__name__}--{kwargs["name"] if len(kwargs)>0 else None} Took {total_time:.4f} seconds\n')
        
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
        sample_book_cleaned = book_df_cleaned.sample(100, random_state=42) #delete

        user_rating_df = pd.read_parquet('data/user_rating_total.parquet')
        user_rating_df_cleaned = user_rating_df.drop_duplicates()
        user_rating_df_cleaned.dropna(inplace=True)
        user_rating_df_cleaned.reset_index(drop = True, inplace = True)
        
        return sample_book_cleaned, user_rating_df_cleaned

    @timeit
    def generate_embeddings(self, df, item, **kwargs):
        if item == 'Description':
            ddata = dd.from_pandas(df[['Id', 'Description']], npartitions=30)
            ddata['Embeddings'] = ddata['Description'].apply(self.embed_text, meta = ('Description', 'object'))
        else:
            ddata = dd.from_pandas(df[['Id', 'Name']], npartitions=30)
            ddata['Embeddings'] = ddata['Name'].apply(self.embed_text, meta = ('Name', 'object'))

        print('======= Computing =======')
        embeddings = ddata.compute()
        print('======= Computing Done =======')
        embeddings_df = pd.DataFrame(embeddings['Embeddings'].tolist())
        embeddings_df.columns = [str(x) for x in range(embeddings_df.shape[1])]
        embeddings_df.insert(0, 'Id', embeddings['Id'].tolist())
        
        return embeddings_df

    @timeit
    def compute_similarity(self, embeddings, **kwargs):
        embeddings_1 = embeddings.iloc[:, 1:]
        embeddings_2 = embeddings.iloc[:, 1:]
        similarity_matrix = embeddings_1 @ embeddings_2.T
        return similarity_matrix

    @timeit
    def generate_user_item_matrix(self, final_similarity_matrix, user_rating_df):
        existing_books = final_similarity_matrix.columns.tolist()
        user_rating_with_existing_books = user_rating_df[user_rating_df['book_id'].isin(existing_books)]
        user_rating_with_existing_books.book_id = user_rating_with_existing_books.book_id.astype('int').astype('str')
        user_rating_with_existing_books.ID = user_rating_with_existing_books.ID.astype('str')

        ddata = dd.from_pandas(user_rating_with_existing_books[['ID', 'book_id', 'Rating']], npartitions=30)
        ddata_aggregated = ddata.groupby(['ID', 'book_id']).aggregate('mean')
        df_aggregated = ddata_aggregated.compute().reset_index()
        user_item_matrix = df_aggregated.pivot(index='ID', columns='book_id', values='Rating').fillna(0)

        return user_item_matrix

    # @timeit
    def get_book_details(self, book_df, book_id):
        """Worker function to get book details."""
        title = book_df[book_df['Id'] == int(book_id)]['Name'].values[0]
        description = book_df[book_df['Id'] == int(book_id)]['Description'].values[0]
        return book_id, title, description

    @timeit
    def get_top_k_recommendations(self, k, filled_matrix, book_df):
        print(f'======= Getting Top {k} Recommendations =======')
        # Prepare the data structure for storing results
        results = []

        # Using ThreadPoolExecutor to parallelize the task
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Create a list of futures
            future_to_user = {executor.submit(self.get_book_details, book_df, book_id): user_id 
                            for user_id in filled_matrix.index 
                            for book_id in filled_matrix.T[user_id].sort_values(ascending=False)[:10].index.tolist()}
            
            # Collecting results as they complete
            for future in concurrent.futures.as_completed(future_to_user):
                user_id = future_to_user[future]
                try:
                    book_id, title, description = future.result()
                    results.append({'user_id': user_id, 'book_id': book_id, 'book_title': title, 'description': description})
                except Exception as exc:
                    print(f'{user_id} generated an exception: {exc}')

        return pd.DataFrame(results)

    @timeit
    def generate_recommendations(self):
        #1. Load data
        print('======= Loading Data =======')
        book_df, user_rating_df = self.load_data()

        #2. Generate embeddings -- Description
        print('======= Generating Embeddings =======')
        print('======= Description =======')
        embeddings_desc_df = self.generate_embeddings(book_df, 'Description', name="description")

        #3. Generate embeddings -- Title
        print('======= Title =======')
        embeddings_title_df = self.generate_embeddings(book_df, 'Title', name="title")

        #4. Item similarity -- item x item matrix
        # Description
        print('======= Generating item similarity -- Desc =======')
        similarity_matrix_desc = self.compute_similarity(embeddings_desc_df, name="description")

        # Title
        print('======= Generating item similarity -- Title =======')
        similarity_matrix_title = self.compute_similarity(embeddings_title_df, name="title")

        # Final Item Similarity Matrix
        print('======= Generating final item similarity =======')
        final_similarity_matrix = 0.4*similarity_matrix_title + 0.6*similarity_matrix_desc
        final_similarity_matrix.columns = embeddings_title_df['Id'].tolist()
        final_similarity_matrix.index = embeddings_title_df['Id'].tolist()
        
        # 5. Generate User Item Matrix (Ratings)
        print('======= Generating user-item matrix =======')
        
        user_item_matrix = self.generate_user_item_matrix(final_similarity_matrix, user_rating_df)

        #6. Generate Recommendation Table -- Predict rating of unrated books
        print('======= Predicting =======')
        final_similarity_matrix.columns = final_similarity_matrix.columns.astype('str')
        final_similarity_matrix.index = final_similarity_matrix.index.astype('str')
        pr = PredictRating()
        filled_matrix = pr.fill_user_item_matrix_parallel(user_item_matrix, final_similarity_matrix, num_processes=4)
       
        #7. Generate Top K Recommendations
        recommendations = self.get_top_k_recommendations(10, filled_matrix, book_df)
        
        recommendations.to_parquet('data/top_k_recommendations_parallel.parquet')

        print('======= Done =======')
        return recommendations

    def index_embedding_vectors(self, data):
        print('===== Upserting =====')
        embed = HuggingFaceEmbeddings(
            model_name='BAAI/bge-large-en-v1.5'
        )

        PINECONE_API = config['PINECONE_API']
        PINECONE_ENV = config['PINECONE_ENV']

        YOUR_API_KEY = PINECONE_API
        YOUR_ENV = PINECONE_ENV

        index_name = 'llm-recommender-system'
        pinecone.init(
            api_key=YOUR_API_KEY,
            environment=YOUR_ENV
        )

        batch_size = 100
        texts = []
        metadatas = []
        data.insert(0, 'category', ['recommended']*data.shape[0])

        index = pinecone.Index(index_name)
        for i in tqdm(range(0, len(data), batch_size)):
            # get end of batch
            i_end = min(len(data), i+batch_size)
            batch = data.iloc[i:i_end]
            # first get metadata fields for this record
            metadatas = [
            {
                'category': record['category'],
                'title': record['book_title'],
                'description': record['description'],
                **({'user_id': str(int(record['user_id']))} if record['category'] == 'recommended' else {})
            }
            
            for _, record in batch.iterrows()]
            # get the list of contexts / documents
            documents = batch['description']
            # create document embeddings
            embeds = embed.embed_documents(documents)
            # get IDs
            ids = batch['book_id'].astype(str)
            # add everything to pinecone
            index.upsert(vectors=zip(ids, embeds, metadatas))
        
        return


if __name__ == '__main__':
    config = dotenv_values(".env")
    rs = RecommendationSystem()
    data = rs.generate_recommendations()
    # data = pd.read_parquet('data/top_k_recommendations_parallel.parquet')
    rs.index_embedding_vectors(data)
    
file_handler.close()