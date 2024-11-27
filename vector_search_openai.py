#built-in packages
import json
import os
import sys

#third-party packages
from dotenv import load_dotenv
import numpy as np
from openai import OpenAI

#own packages
sys.path.append(r"C:\Users\Andres.DESKTOP-D77KM25\OneDrive - Stanford\Laboral\Lawgorithm\Repos\3.Embedding")
from openai_functions import get_embedding
from embedding import download_blob_content

#load the .env file
load_dotenv()

#import variables
openai_key=os.getenv('openai_key')

client = OpenAI(api_key=openai_key)


# Step 1: Get user query
def get_user_query():
    """
    Gets the user's query as input.
    """
    query = input("Enter your query: ")
    return query


# Step 3: Calculate similarity
def calculate_similarity(query_embedding, database_embeddings):
    """
    Calculates similarity between the query embedding and the database embeddings.
    
    Args:
        query_embedding (np.ndarray): The embedding of the query.
        database_embeddings (np.ndarray): An array of embeddings in the database.
        
    Returns:
        np.ndarray: Similarity scores for each database entry.
    """
    # Using cosine similarity
    norms = np.linalg.norm(database_embeddings, axis=1) * np.linalg.norm(query_embedding)
    similarity = np.dot(database_embeddings, query_embedding) / (norms + 1e-9)
    return similarity


# Step 4: Retrieve top-k results
def retrieve_top_k(database, similarities, k=5):
    """
    Retrieves the top-k entries from the database based on similarity scores.
    
    Args:
        database (list): The database of entries.
        similarities (np.ndarray): The similarity scores.
        k (int): The number of top entries to retrieve.
        
    Returns:
        list: Top-k entries from the database.
    """
    top_k_indices = np.argsort(similarities)[::-1][:k]
    return [database[i] for i in top_k_indices]


# Step 5: Create prompt
def create_prompt(retrieved_sources):
    """
    Creates a prompt based on the retrieved sources.
    
    Args:
        retrieved_sources (list): List of top-k sources retrieved from the database.
        
    Returns:
        str: A concatenated string that forms the prompt.
    """
    prompt = "\n\n".join(retrieved_sources)
    return f"Relevant sources for your query:\n\n{prompt}"


def main1():
    # Mock embedding model and database
    def mock_embedding_model(text):
        # Simple example, returns an array of word lengths for demonstration
        return np.array([len(word) for word in text.split()])

    database = ["This is the first document.", 
                "Here is another relevant source.", 
                "A third document with relevant content.",
                "More information in this entry.",
                "Final document in the database."]
    
    database_embeddings = np.array([mock_embedding_model(doc) for doc in database])

    # Pipeline
    query = get_user_query()
    query_embedding = embed_user_query(query, mock_embedding_model)
    similarity = calculate_similarity(query_embedding, database_embeddings)
    retrieved_sources = retrieve_top_k(database, similarity, k=5)
    prompt = create_prompt(retrieved_sources)

    print(prompt)
    

def main2():
    query = get_user_query()
    query = get_embedding(query)
    database =  download_blob_content()






# Example usage
if __name__ == "__main__":
    main1()