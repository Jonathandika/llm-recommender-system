# LLM Recommender System

## Abstract
This project examines the efficacy of Large Language Models (LLMs) integrated with parallel computing to optimize recommendation systems. With a focus on Retrieval Augmented Generation (RAG) models, the research assesses the impact of parallel processing on the speed and relevance of book recommendations. Methodologically, the study contrasts the performance of traditional sequential processing against a parallelized approach across key operations such as data preprocessing, embedding generation, similarity computation, and recommendation prediction. Results indicate that parallel computing significantly decreases operational times by up to 92.89%, enhancing the system's efficiency. The practical application of the RAG model in a chatbot interface confirms the model's capability to deliver personalized and contextually appropriate book suggestions. These findings highlight the potential of integrating parallel processing with LLMs to advance the responsiveness and accuracy of content recommendation systems.

## Features
- **Parallel Computing:** Enhances the efficiency of data processing operations, significantly reducing the time required for tasks such as data preprocessing, embedding generation, and similarity computation.
- **RAG Model:** Utilizes Retrieval Augmented Generation to deliver personalized and contextually appropriate book recommendations.
- **Scalability:** Designed to handle large datasets efficiently through parallel processing techniques.
- **Chatbot Integration:** Provides a practical application of the recommendation system in a chatbot interface.

## Variants

1. **Content-Based Recommendation System**  
   This variant utilizes a content-based recommendation approach to tailor book suggestions according to individual user preferences. By employing matrix multiplication, the algorithm calculates similarity scores between user preferences and available items, streamlining the process to deliver more accurate recommendations. The recommendation generation is parallelized to enhance efficiency, significantly reducing processing time. This variant can be launched by running Streamlit on `app.py` using the following command:

   ```bash
   streamlit run app.py

2. **Reinforcement Learning Recommendation System**  
   This variant leverages reinforcement learning, specifically a Double Deep Q-Network (DDQN) model ([DDQN paper](https://arxiv.org/abs/1509.06461)), to refine recommendations based on user feedback. After each recommendation interaction, user feedback is evaluated—whether positive or negative—using a Large Language Model (LLM) to interpret and score the feedback. This feedback is then used to update the reinforcement learning model, allowing it to learn and adapt to the user’s preferences over time. As the model gains experience, it becomes more effective at aligning recommendations with the unique preferences of each user, resulting in a personalized and adaptive recommendation experience. This variant can be launched by running Streamlit on `app_RL.py` using the following command:

   ```bash
   streamlit run app_RL.py
