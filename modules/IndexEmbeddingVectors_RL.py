import pandas as pd
import numpy as np
from dotenv import dotenv_values
import pinecone
from langchain.embeddings import HuggingFaceEmbeddings
from tqdm import tqdm

class IndexEmbeddingVectors:
    def __init__(self):
        self.config = dotenv_values(".env")
        pass

    def index_embedding_vectors(self, data, list_type):
        print('===== Upserting =====')
        embed = HuggingFaceEmbeddings(
            model_name='BAAI/bge-large-en-v1.5'
        )

        PINECONE_API = self.config['PINECONE_API']
        PINECONE_ENV = self.config['PINECONE_ENV']

        YOUR_API_KEY = PINECONE_API
        YOUR_ENV = PINECONE_ENV

        index_name = 'rl-llm-recsys'
        pinecone.init(
            api_key=YOUR_API_KEY,
            environment=YOUR_ENV
        )

        batch_size = 100
        metadatas = []

        data.insert(0, 'category', [f'{list_type}']*data.shape[0])
        # print(data)

        index = pinecone.Index(index_name)
        for i in tqdm(range(0, len(data), batch_size)):
            # get end of batch
            i_end = min(len(data), i+batch_size)
            batch = data.iloc[i:i_end]
            # first get metadata fields for this record
            metadatas = [
            {
                'category': record['category'],
                'book_id': record['book_id'],
                'book_title': record['book_title'],
                'description': record['description'],
                **({'user_id': (int(record['user_id']))} if record['category'] == 'recommended' else {}),
                **({'user_encoded': (int(record['user_encoded']))} if record['category'] == 'recommended' else {})

            }
            
            for _, record in batch.iterrows()]
            # get the list of contexts / documents
            documents = batch['description']
            # create document embeddings
            embeds = embed.embed_documents(documents)
            # get IDs
            ids = batch.index.astype(str)
            # add everything to pinecone
            index.upsert(vectors=zip(ids, embeds, metadatas))
        
        return
    
    def delete_vectors(self, user_id):
        PINECONE_API = self.config['PINECONE_API']
        PINECONE_ENV = self.config['PINECONE_ENV']

        YOUR_API_KEY = PINECONE_API
        YOUR_ENV = PINECONE_ENV

        index_name = 'rl-llm-recsys'
        pinecone.init(
            api_key=YOUR_API_KEY,
            environment=YOUR_ENV
        )

        index = pinecone.Index(index_name)

        res = index.query(
            vector=[ 0.0 for i in range(1024)],
            filter={
                "user_id": user_id,
            },
            top_k=5,
            include_metadata=True
        )

        old_books_ids = [match['id'] for match in res['matches']]

        index.delete(ids=old_books_ids)
        return

if __name__ == '__main__':
    config = dotenv_values(".env")
    print('getting data')
    book_df = pd.read_parquet('data/book_eng.parquet')
    book_df_cleaned = book_df.dropna(subset = ['Description'])
    book_df_cleaned.reset_index(drop = True, inplace = True)
    book_df_cleaned.RatingDistTotal = book_df_cleaned.RatingDistTotal.str.replace('total:', '').astype(int)
    book_df_cleaned.drop_duplicates(subset=['Name'], keep='first', inplace = True)
    top_100_books = book_df_cleaned.sort_values('RatingDistTotal', ascending = False)[:100]
    final_top_100_books = top_100_books[['Id', 'Name', 'Description']]
    final_top_100_books = final_top_100_books.rename(columns = {'Id': 'book_id', 'Name': 'book_title', 'Description':'description'})
    i = IndexEmbeddingVectors()
    i.index_embedding_vectors(final_top_100_books, 'popular')