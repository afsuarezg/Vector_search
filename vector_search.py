import json
import sys
import torch
from transformers import AutoTokenizer, AutoModel





def embedd_query(text, huggingface_model):
    """
    Embedd the query using Huggingface models.
    
    Args:
    text (str): The input text to be embedded.
    huggingface_model (str): The name of the Huggingface model to use.
    
    Returns:
    torch.Tensor: The embedding of the input text.
    """
    tokenizer = AutoTokenizer.from_pretrained(huggingface_model)
    model = AutoModel.from_pretrained(huggingface_model)
    inputs = tokenizer(text, return_tensors='pt')
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)


def call_database(path):
    """
    Import the json file from Blob Storage containing the legal text data and its embeddings.
    
    Args:
    path (str): The path to the json file.
    
    Returns:
    dict: The database containing legal text data and its embeddings.
    """
    with open(path, 'r') as file:
        database = json.load(file)
    return database


def calculate_similarity(query_embedding, database):
    """
    Calculate the similarity between the query and the embeddings of the sources contained in the database.
    
    Args:
    query_embedding (torch.Tensor): The embedding of the query.
    database (dict): The database containing legal text data and its embeddings.
    
    Returns:
    list: A list of similarity scores.
    """
    similarities = []
    for entry in database['entries']:
        source_embedding = torch.tensor(entry['embedding'])
        similarity = torch.nn.functional.cosine_similarity(query_embedding, source_embedding.unsqueeze(0))
        similarities.append(similarity.item())
    return similarities


def retrieve_top_k(database, similarities, k=5):
    """
    Using the similarities previously calculated, extract the text of the k observations with the highest similarity score.
    
    Args:
    database (dict): The database containing legal text data and its embeddings.
    similarities (list): A list of similarity scores.
    k (int): The number of top observations to retrieve.
    
    Returns:
    list: A list of the top k legal texts.
    """
    top_k_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)[:k]
    top_k_texts = [database['entries'][i]['text'] for i in top_k_indices]
    return top_k_texts


def create_prompt(top_k_texts):
    """
    Create a prompt that passes the text of the sources previously identified and returns a string.
    
    Args:
    top_k_texts (list): A list of the top k legal texts.
    
    Returns:
    str: The generated prompt.
    """
    prompt = "The following are the most relevant legal texts:\n\n"
    for i, text in enumerate(top_k_texts):
        prompt += f"{i+1}. {text}\n\n"
    return prompt


if __name__ == '__main__':
     pass


