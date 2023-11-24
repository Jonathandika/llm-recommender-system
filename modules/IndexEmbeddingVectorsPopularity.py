import pandas as pd
import numpy as np
from dotenv import dotenv_values
import pinecone
from langchain.embeddings import HuggingFaceEmbeddings
from tqdm import tqdm

def index_embedding_vectors(data):
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
    metadatas = []

    data.columns = ['book_id', 'book_title', 'description']
    data.insert(0, 'category', ['popular']*data.shape[0])
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
            'title': record['book_title'],
            'description': record['description']
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

config = dotenv_values(".env")
book_df = pd.read_parquet('data/book_eng.parquet')
book_df_cleaned = book_df.dropna(subset = ['Description'])
book_df_cleaned.reset_index(drop = True, inplace = True)
book_df_cleaned.RatingDistTotal = book_df_cleaned.RatingDistTotal.str.replace('total:', '').astype(int)
top_10_books = book_df_cleaned.sort_values('RatingDistTotal', ascending = False)[:1000]
index_embedding_vectors(top_10_books[['Id', 'Name', 'Description']])