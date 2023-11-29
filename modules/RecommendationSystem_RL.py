import random
import argparse
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras import losses
import itertools as it
import pandas as pd
import keras
from keras import backend as K
from collections import Counter
import heapq
from tqdm import tqdm
import tensorflow as tf

from helper.DQNAgent import DQNAgent

parser = argparse.ArgumentParser()

parser.add_argument('--Model_Type', type=int, default=1, help='0 = Initial Model & 1 = Target Model')
parser.add_argument('--State_Size', type=int, default=2, help='number of articles previously read by a user')
parser.add_argument('--batch', type=int, default=5, help='input batch size')
parser.add_argument('--niter', type=int, default=100, help='number of epochs to train for per user')
parser.add_argument('--LR', type=float, default=0.001, help='learning rate')
parser.add_argument('--deque_length', type=int, default=2000, help='memory to remember')
parser.add_argument('--discount', type=float, default=0.95, help='gamma')
parser.add_argument('--eps', type=float, default=1.0, help='epsilon')
parser.add_argument('--eps_decay', type=float, default=0.995, help='epsilon decay')
parser.add_argument('--eps_min', type=float, default=0.01, help='minimum epsilon')

optim = parser.parse_args(args = [])

class RecommendationSystemRL:
    """
    This class contains all the functions for the Recommendation System.
    """

    train_data = 'data/RL/traindata.csv'
    states = 'data/RL/TFIDF-States.csv'
    books = 'data/RL/booksamples.csv'

    def __init__(self, retrain=False):
        self.train_data, self.states, self.books, self.user_ids, self.book_names = self._load_data(self.train_data, self.states, self.books)
        self.Encoded, self.Original_States = self._process_combination(self.states, self.book_names, optim.State_Size)
        self.batch_size = optim.batch
        self.state_size = len(self.Encoded[0])
        self.action_size = len(self.book_names)

        if not retrain:
            self.agent = DQNAgent(self.state_size, self.action_size, retrain=False)
            self.rec_initial_full = pd.read_csv('output/RL/rec_initial.csv')
            
        else:
            self.agent = DQNAgent(self.state_size, self.action_size, retrain=True)            
            self.rec_initial_full = None

        self.Index = []


    def _load_data(self, train_data, states, books):
        train_data_df = pd.read_csv(train_data)
        states_df = pd.read_csv(states)
        books_df = pd.read_csv(books)

        user_ids = train_data_df['user_encoded'].to_list()
        book_names = books_df['Name']

        return train_data_df, states_df, books_df, user_ids, book_names
    
    def _process_combination(self, States, book_names, state_size = 2):
        """
        Making the states

        Parameters:
        States, book_names list, and state_size = 2

        Returns:
        df: data with books ID

        """
        State_Space = []
        Temp_List = []
        for i in range(0,len(States)):
            Temp_List.append(States.iloc[i:i+1].values.tolist())
        Temp_List.pop(0)
        for i in range(len(Temp_List)):
            State_Space.append(Temp_List[i][0])
        Encoded = [list(map(float, x)) for x in State_Space]
        Encoded_States = []
        Encoded_States.append(Encoded)

        #making combination of 2 books together (kyk yg buat tf-idf)
        Original_States = []
        Original_States.append(list(it.combinations(book_names,state_size)))
        return Encoded, Original_States
    
    def _new_state(self, S, action_idx):
        """
        Creates a new state from existing state and action.

        Parameters:
        arg1 (list): Current State
        arg2 (int): Action
        Returns:
        list: Returns the new state.

        """
        New_State=(S[1], self.book_names[action_idx])
        return New_State
    
    def _state_index(self, S):
        """
        Finds out the index of the current state.

        Parameters:
        arg1 (list): State

        Returns:
        int: Returns the index.

        """
        Secondary_State=[]
        Secondary_State=list(it.permutations(S, len(S)))

        for i in range(0,len(Secondary_State)):
            if Secondary_State[i] in self.Original_States[0]:
                self.Index.append(self.Original_States[0].index(Secondary_State[i]))
                break

        return(self.Index[-1])
    
    def _copy_model(self, model):
        """
        Saves and Loads the Model as Target Model.

        Parameters:
        arg1 (model): Initial Model

        Returns:
        model: Returns the Target Model.

        """
        model.save('tmp_model')
        target_model = keras.models.load_model('tmp_model')
        return target_model
    
    def _add_history(self, data):
        """
        Creates history of a a combination of 2 'recent' books that have been read for each user in the df.

        Parameters:
        arg1 (df): training data

        Returns:
        df: training data with history

        """
        historys = []
        user_ids = data['ID'].unique()

        for user_id in user_ids:
            user_data = data[data['ID'] == user_id]
            books_read = user_data.head(2)[['Name']].values.tolist()

            books_read= [item[0] for item in books_read]
            historys.append(books_read)
        historydf = pd.DataFrame({'ID' : user_ids, 'history' : historys})
        data = pd.merge(data,historydf, on='ID')
        return data
    
    def _recommend2(self, model_type, Test_States):
        """
        Recommends articles for every possible state.

        Parameters:
        arg1 (model): Type of Model - Target Model / Initial Model

        Returns:
        list: Returns Recommendation Scores for each article for a state.
        """

        Recommendation_Scores = model_type.predict(Test_States)
        Scores = heapq.nlargest(100, range(len(Recommendation_Scores[0])), key=Recommendation_Scores[0].__getitem__)
        Scores = [x + 1 for x in Scores]
        return Scores, Recommendation_Scores
    
    def _add_book_ids(self, df, df_books):
        """
        Merges the book_ids for the recommendation dfs

        Parameters:
        arg1 (df): dataframe, books ID mapping dataframe

        Returns:
        df: data with books ID

        """

        if 'Name' not in df_books.columns:
            raise ValueError("Column 'Name' not found in df_books")

        df_merged = df.copy()

        for i in range(1, 6):
            rec_col = f'rec_{i}'
            df_merged = pd.merge(df_merged, df_books, left_on=[rec_col], right_on=['Name'], how='left', suffixes=('', f'_temp'))
            df_merged = df_merged.rename(columns={'Id': f'recid_{i}'})
            
            # Drop 'Name_temp' if it exists
            if f'Name_temp' in df_merged.columns:
                df_merged = df_merged.drop([f'Name_temp'], axis=1)
            
            if f'Name' in df_merged.columns:
                df_merged = df_merged.drop([f'Name'], axis=1)

        return df_merged


    
    def get_recommended_items(self):
        model_type = self.agent.model

        # Get the recommendation using Double Deep Q-Learning
        books_rec = []
        historys = []
        top_books = []
        history_books = []

        train_data_history = self._add_history(self.train_data)
        s_list = train_data_history['history'].drop_duplicates().tolist()

        for i in tqdm(range(0, len(s_list))):
            S = s_list[i]
            state_id = self._state_index(S)
            S1 = self.Encoded[i]
            S1 = np.reshape(S1, [1, self.state_size])
            Score_List, Recommendation_Score = self._recommend2(model_type, S1)
            top_books = [self.book_names[book_id-1] for book_id in Score_List]
            history_books = S
            books_rec.append(top_books)
            historys.append(history_books)

        recommended = pd.DataFrame({'recommended': books_rec, 'history': historys})
        train_data_history = self._add_history(self.train_data)
        train_data_history['history'] = train_data_history['history'].apply(lambda x: tuple(sorted(x)))
        recommended['history'] = recommended['history'].apply(lambda x: tuple(sorted(x)))
        books_idname = self.books[['Id', 'Name']]
        recommended_final = pd.merge(train_data_history, recommended, on='history')

        user_books_read = recommended_final.groupby('user_encoded')['Name'].apply(set)
        rec_1 = []
        rec_2 = []
        rec_3 = []
        rec_4 = []
        rec_5 = []

        for _, row in recommended_final.iterrows():
            user_id = row['user_encoded']
            recs = row['recommended']

            recs_unread = [rec for rec in recs if rec not in user_books_read[user_id]]
            rec_1.append(recs_unread[0] if recs_unread else None)
            rec_2.append(recs_unread[1] if len(recs_unread) > 1 else None)
            rec_3.append(recs_unread[2] if len(recs_unread) > 2 else None)
            rec_4.append(recs_unread[3] if len(recs_unread) > 3 else None)
            rec_5.append(recs_unread[4] if len(recs_unread) > 4 else None)

        recommended_final['rec_1'] = rec_1
        recommended_final['rec_2'] = rec_2
        recommended_final['rec_3'] = rec_3
        recommended_final['rec_4'] = rec_4
        recommended_final['rec_5'] = rec_5

        recommended_final_clean = recommended_final[['ID','user_encoded', 'rec_1', 'rec_2','rec_3', 'rec_4', 'rec_5']]
        recommended_final_clean.drop_duplicates(inplace= True)
        recommended_final_clean = self._add_book_ids(recommended_final_clean, books_idname)
        return recommended_final, recommended_final_clean
    
    def get_recommended_items2(self, userid, book_accepted):

        model_type = self.agent.model

        old_book = self.train_data.groupby('user_encoded')['Name'].first().loc[userid]
        new_book = self.train_data.loc[self.train_data['book_id'] == book_accepted, 'Name'].values[0]
        S = [old_book,new_book]

        books_rec = []
        historys = []
        top_books = []
        history_books = []

        state_id = self._state_index(S)
        S1 = self.Encoded[state_id]
        S1 = np.reshape(S1, [1, self.state_size])
        books_idname = self.books[['Id', 'Name']]

        # Get the recommendation using Double Deep Q-Learning
        Score_List, Recommendation_Score = self._recommend2(model_type, S1)
        top_books = [self.book_names[book_id-1] for book_id in Score_List]
        history_books = S
        books_rec.append(top_books)
        historys.append(history_books)
        actualid = self.train_data[self.train_data['user_encoded'] == userid]['ID'].unique()
        recommended = pd.DataFrame({'ID' : actualid ,'Name' : new_book, 'user_encoded' : userid, 'recommended': books_rec, 'history': historys})
        recommended['history'] = recommended['history'].apply(lambda x: tuple(sorted(x)))
        user_books_read = self.train_data.groupby('user_encoded')['Name'].apply(set)
        user_books_read[userid].add(new_book)

        rec_1 = []
        rec_2 = []
        rec_3 = []
        rec_4 = []
        rec_5 = []

        for _, row in recommended.iterrows():
            user_id = row['user_encoded']
            recs = row['recommended']

            recs_unread = [rec for rec in recs if rec not in user_books_read[user_id]]
            rec_1.append(recs_unread[0] if recs_unread else None)
            rec_2.append(recs_unread[1] if len(recs_unread) > 1 else None)
            rec_3.append(recs_unread[2] if len(recs_unread) > 2 else None)
            rec_4.append(recs_unread[3] if len(recs_unread) > 3 else None)
            rec_5.append(recs_unread[4] if len(recs_unread) > 4 else None)

        recommended['rec_1'] = rec_1
        recommended['rec_2'] = rec_2
        recommended['rec_3'] = rec_3
        recommended['rec_4'] = rec_4
        recommended['rec_5'] = rec_5

        old_rec_df = self.rec_initial_full[self.rec_initial_full['user_encoded'] != userid]
        recommended_final = pd.concat([old_rec_df, recommended], axis=0, ignore_index=True)
        recommended_final_clean = recommended_final[['ID','user_encoded', 'rec_1', 'rec_2','rec_3','rec_4','rec_5']]
        recommended_final_clean.drop_duplicates(inplace= True)
        recommended_final_clean = self._add_book_ids(recommended_final_clean, books_idname)
        return recommended_final, recommended_final_clean

    
    def get_initial_recommendation(self):
       
        user_ids = np.unique(self.user_ids)

        # Iterate over every user
        for i in range(len(user_ids)):
            print("=============================================================================")
            print("For User", user_ids[i])

            # Iterate to train the model for every state of the current user
            for j in range(len(self.train_data['user_encoded'])):
                if  user_ids[i] == self.train_data['user_encoded'][j]:

                    # getting the state
                    S = [self.train_data['Name'][j], self.train_data['Name'][j+1]]
                    print(S)

                    # reading user csv
                    x = (user_ids[i]).__str__()
                    user = pd.read_csv('data/RL/user_' + x + '.csv')
                    done = False
                    flag = 0

                    # Create a list -> Read having all the books the user has ignored/read in a particular state
                    read = list(S)
                    e=0

                    while e<optim.niter:
                        print('iter' , e)
                        state_id = self._state_index(S) # Find the index of the initial state
                        S1 = self.Encoded[state_id] # Find the index of the one-hot encoded state
                        S1 = np.reshape(S1, [1, self.state_size])
                        action = self.agent.act(S1) # Perform a recommendation
                        print('action ', action)
                        if user['action'][action] == 1:
                            user_action = 1
                        else:
                            user_action = 0

                        if(self.book_names[action] in read):
                            print('read, continue')
                            continue

                        # Assign Rewards for each action and change the user state accordingly
                        # The state of the user only changes if he has clicked on the recommendation

                        if user_action == 1:
                            reward = 15
                            print("Epoch" , e)
                            print("User Clicked")
                            print("Book ID", action+1)
                            print("-----------------------------------------------------------------------------")
                            done= True
                            next_state=self._new_state(S,action)
                            read = list(next_state)
                        elif user_action == 0:
                            print('reward decreased')
                            reward = -5
                            read.append(self.book_names[action])
                            next_state=self.Original_States[0][state_id]

                        next_state_index = self._state_index(next_state)
                        Encoded_Next_State = self.Encoded[next_state_index]
                        Encoded_Next_State = np.reshape(Encoded_Next_State, [1, self.state_size])

                        # Store the states, actions and rewards in a double queue
                        self.agent.remember(S1, action, reward, Encoded_Next_State, done)

                        S = list(next_state)
                        print('new S', S)

                        # Start training the model once the memory has more actions than the batch-size
                        if len(self.agent.memory) > self.batch_size:
                            # For every 'N' save the model and load it as Target-Model keeping the Targeted Q-Value constant for 'N' iterations
                            if e%100 == 0:
                                print("Target Model Saved")
                                target_model = self._copy_model(self.agent.model)
                                flag = 1
                            if flag == 1:
                                # Train using Target-Model
                                self.agent.replay2(self.batch_size, target_model)
                            # Train using Regular Model
                            self.agent.replay(self.batch_size)

                        # Stop the Training Process if length of read books is equal to the number of books
                        if len(read) == len(self.books)-2:
                            break
                        e=e+1
                    break
        rec_initial_full, rec_initial = self.get_recommended_items()

        self.rec_initial_full = rec_initial_full

        return rec_initial
    
    def get_new_recommendation(self, userid, book_accepted):
        print("=============================================================================")
        print("For User", userid)
        for j in range(len(self.train_data['user_encoded'])):
            if  userid == self.train_data['user_encoded'][j]:

                # getting the state
                old_book = self.train_data['Name'][j]
                new_book = self.train_data.loc[self.train_data['book_id'] == book_accepted, 'Name'].values[0]
                S = [old_book,new_book]
                print(S)

                # reading user csv
                x = (userid).__str__()
                user = pd.read_csv('data/RL/user_' + x + '.csv')
                user.loc[user['book_id'] == book_accepted, 'action'] = 1
                # Initially the user has not Clicked, hence done = False
                done = False
                flag = 0
                read = list(S)
                e=0

                while e<1:
                    print('iter' , e)
                    state_id = self._state_index(S) # Find the index of the initial state
                    S1 = self.Encoded[state_id] # Find the index of the one-hot encoded state
                    S1 = np.reshape(S1, [1, self.state_size])
                    action = self.agent.act(S1) # Perform a recommendation
                    print('action ', action)
                    if user['action'][action] == 1:
                        # print('user action:',user_action)
                        user_action = 1
                    else:
                        user_action = 0


                    if(self.book_names[action] in read):
                        print('read, continue')
                        continue

                    # Assign Rewards for each action and change the user state accordingly
                    # The state of the user only changes if he has clicked on the recommendation
                    if user_action == 1:
                        reward = 15
                        print("Epoch" , e)
                        print("User Clicked")
                        print("Book ID", action+1)
                        print("-----------------------------------------------------------------------------")
                        done= True
                        next_state=self._new_state(S,action)
                        read = list(next_state)
                    elif user_action == 0:
                        print('reward decreased')
                        reward = -15
                        read.append(self.book_names[action])
                        next_state=self.Original_States[0][state_id]
                    next_state_index = self._state_index(next_state)
                    Encoded_Next_State = self.Encoded[next_state_index]
                    Encoded_Next_State = np.reshape(Encoded_Next_State, [1, self.state_size])

                    # Store the states, actions and rewards in a double queue
                    self.agent.remember(S1, action, reward, Encoded_Next_State, done)

                    S = list(next_state)
                    print('new S', S)

                    # Start training the model once the memory has more actions than the batch-size
                    if len(self.agent.memory) > self.batch_size:
                        # For every 'N' save the model and load it as Target-Model keeping the Targeted Q-Value constant for 'N' iterations
                        if e%100 == 0:
                            print("Target Model Saved")
                            target_model = self._copy_model(self.agent.model)
                            flag = 1
                        if flag == 1:
                            # Train using Target-Model
                            self.agent.replay2(self.batch_size, target_model)
                        # Train using Regular Model
                        self.agent.replay(self.batch_size)

                    # Stop the Training Process if length of read books is equal to the number of books
                    if len(read) == len(self.books)-2:
                        break
                    e=e+1
                break
        newrecfull, newrec = self.get_recommended_items2(userid, book_accepted)

        self.rec_initial_full = newrecfull

        return newrec
    
    

if __name__ == "__main__":
    # rs = RecommendationSystemRL(retrain=True)
    # rec_init = rs.get_initial_recommendation()
    # rec_init.to_csv('output/RL/rec_initial.csv', index=False)

    # New Rec
    rs = RecommendationSystemRL(retrain=False)
    rec_new = rs.get_new_recommendation(1, 1076302)
    rec_new.to_csv('output/RL/rec_new.csv', index=False)