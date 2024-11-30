#built-in packages
import json
import os
import sys

#third-party packages
from dotenv import load_dotenv
import numpy as np
from openai import OpenAI
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import torch

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


def get_top_k_indices(tensor, k):
    """
    Returns the indices of the k elements with the higher values in a tensor.

    Args:
        tensor: The input tensor.
        k: The number of top elements to return.

    Returns:
        A list of indices of the top k elements.
    """
    # Get the indices of the top k elements
    top_k_indices = torch.topk(tensor, k).indices.tolist()
    return top_k_indices


def get_contents_from_indices(data, indices_list, column):
    """
    Returns the contents in the given indices from a column in a pandas DataFrame.

    Args:
        data: The pandas DataFrame.
        indices_list: A list of indices.
        column_name: The name of the column to retrieve contents from.

    Returns:
        A list of contents corresponding to the given indices.
    """
    contents = []
    for indices in indices_list:
        for index in indices:
            contents.append(data.iloc[index][column])  # Replace 'text' with your desired column name
    return contents


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
    return f"Las fuentes legales relevantes para su consulta son:\n\n{prompt}"


def main1():
    query_embedding = get_embedding(get_user_query())

    database =  download_blob_content(account_url="https://lawgorithm.blob.core.windows.net", 
                            container_name='jurisprudencia-embeddings', 
                            blob_name='jurisprudencia-embeddings-2023.json')

    # Convert the bytes object to a string
    database_str = database.decode('utf-8')

    # Load the string as a list of dictionaries
    database_list = json.loads(database_str)

    # Convert the list of dictionaries to a pandas dataframe
    database = pd.DataFrame(database_list)

    database = database.iloc[[i for i,elem in enumerate(database['embedding'].to_list()) if elem != [None]]]

    similarities = torch.from_numpy(cosine_similarity([query_embedding], database['embedding'].to_list(), dense_output=True))

    sources=  get_contents_from_indices(database,  get_top_k_indices(similarities, 6), 'text')

    prompt = create_prompt(sources)
    print(prompt)   


if __name__ == "__main__":
    main1()