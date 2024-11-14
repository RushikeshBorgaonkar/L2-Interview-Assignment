import os
from db_config import DatabaseManager
from embedding_generator import EmbeddingGenerator
from groq import Groq
from langchain_community.document_loaders import PyPDFLoader
from typing import List

DATA_FILE_PATHS = ['PEPSI-2022-Presentation1.pdf', 'q1-2022-pep_transcript.pdf'] 

def load_text_samples(file_paths):
    texts = []
    for file_path in file_paths:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file '{file_path}' not found.")
        
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        
        for doc in documents:
            texts.append(doc.page_content)
            print(f"Extracted from {file_path}: {doc.page_content}")
    
    return list(dict.fromkeys(filter(None, texts)))

def generate_augmented_response(query: str, retrieved_items: List[tuple[str, float]]) :
    
    context = "\n\n".join(f"Document {idx + 1}:\n{text}" 
                         for idx, (text, _) in enumerate(retrieved_items))
    
    prompt = f"""Please provide a comprehensive response to the question using information from ALL the provided documents below. 
    Context Documents:
    {context}

    Question: {query}
    Please provide a detailed response incorporating information from all documents.
    """

    client = Groq(api_key=os.getenv('GROQ_API_KEY'))
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system", 
                "content": """You are an assistant that provides comprehensive answers by analyzing and 
                synthesizing information from multiple documents. Always strive to include relevant 
                information from all provided documents in your response."""
            },
            {"role": "user", "content": prompt}
        ],
        model="llama3-8b-8192",
        temperature=0.3,  
        max_tokens=1024   
    )

    response = chat_completion.choices[0].message.content.strip()
    
    # print(f"Generated Response: {response}")
    
    return {
        "query": query,
        "generated_response": response
    }

def main():
    query_text = input("Enter your query: ")
    db_manager = DatabaseManager()
    embedding_gen = EmbeddingGenerator()

    db_manager.clear_embeddings()

    texts = load_text_samples(DATA_FILE_PATHS)
    processed_texts = set()
    for idx, text in enumerate(texts):
        if text in processed_texts:
            continue
        processed_texts.add(text)
        
        embedding = embedding_gen.generate_embedding(text)
        db_manager.add_embedding_to_db(embedding, text_id=str(idx), text_content=text)
        # print(f"Added embedding for: {text}")

    query_embedding = embedding_gen.generate_embedding(query_text)
    similar_items = db_manager.search_similar_vectors(query_embedding, top_k=2)

    result = generate_augmented_response(query_text, similar_items)
    # print(result)
    db_manager.close()
    
    print(f"\nQuery Result:\nQuery: {result['query']}")
    print(f"\nGenerated Response:\n{result['generated_response']}")
   

if __name__ == "__main__":
    main()
