import pandas as pd
import numpy as np
from multiprocessing import Pool

class PredictRating():
    def __init__(self):
        pass

    def predict_rating(self, user_id, item_id, user_item_df, item_similarity_df):
        user_ratings = user_item_df.loc[user_id]

        item_similarity = item_similarity_df[item_id]

        similarities = item_similarity[user_ratings.index]

        weighted_ratings = user_ratings * similarities
        prediction = weighted_ratings.sum()

        return prediction

    def fill_user_ratings(self, user_id, user_item_df, item_similarity_df):
        user_ratings = user_item_df.loc[user_id].copy()
        for item_id in user_item_df.columns:
            if user_ratings[item_id]==0:
                user_ratings[item_id] = self.predict_rating(user_id, item_id, user_item_df, item_similarity_df)

        return user_id, user_ratings

    def fill_user_item_matrix_parallel(self, user_item_df, item_similarity_df, num_processes=None):
        with Pool(processes=num_processes) as pool:
            # Create a list of tasks (each task is a user ID)
            tasks = [(user_id, user_item_df, item_similarity_df) for user_id in user_item_df.index]

            # Distribute the tasks across the processes and collect the results
            results = pool.starmap(self.fill_user_ratings, tasks)

        pool.close()
        pool.join()
        # Reconstruct the filled user-item matrix
        filled_user_item_df = pd.DataFrame(index=user_item_df.index, columns=user_item_df.columns)
        for user_id, user_ratings in results:
            filled_user_item_df.loc[user_id] = user_ratings
        
        rated_loc = np.where(user_item_df > 0)
        for row, col in zip(rated_loc[0], rated_loc[1]):
            filled_user_item_df.iat[row,col] = -1

        return filled_user_item_df

    def fill_user_item_matrix_nonparallel(self, user_item_df, item_similarity_df):
        filled_user_item_df = pd.DataFrame(index=user_item_df.index, columns=user_item_df.columns)
        for user_id in user_item_df.index:
            _, user_ratings = self.fill_user_ratings(user_id, user_item_df, item_similarity_df)
            filled_user_item_df.loc[user_id] = user_ratings

        
        rated_loc = np.where(user_item_df > 0)
        for row, col in zip(rated_loc[0], rated_loc[1]):
            filled_user_item_df.iat[row,col] = -1

        return filled_user_item_df
