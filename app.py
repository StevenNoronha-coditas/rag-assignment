from sentence_transformers import SentenceTransformer, CrossEncoder
import numpy as np
import psycopg2
import faiss
import PyPDF2
from groq import Groq

# Load the embedding and cross-encoder re-ranking models
embedding_model = SentenceTransformer('all-mpnet-base-v2')  # Switch to a larger, more accurate model
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')  # Cross-encoder for re-ranking

# Load PDF and extract text
def extract_text_from_pdf(pdf_path):
    text_chunks = []
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + " "
        
        # Create more meaningful chunks with increased overlap
        words = text.split()
        chunk_size = 200  # number of words per chunk
        overlap = 100     # increased overlap for better context capture
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            if len(chunk.strip()) > 0:
                text_chunks.append(chunk)
    return text_chunks

# Create embeddings and store in database
def store_embeddings(chunks):
    connection = psycopg2.connect(
        host="172.25.170.149",
        port="5432",
        dbname="postgres",
        user="postgres",
        password="steven123"
    )
    cursor = connection.cursor()
    
    # Create table if not exists
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id SERIAL PRIMARY KEY,
            content TEXT,
            embedding vector(768)
        );
        CREATE INDEX IF NOT EXISTS documents_embedding_idx ON documents 
        USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100);
    """)
    
    # Check if there are any existing records
    cursor.execute("SELECT COUNT(*) FROM documents")
    count = cursor.fetchone()[0]
    
    if count == 0:
        print("No existing embeddings found. Generating new embeddings...")
        for chunk in chunks:
            # Generate embedding
            embedding = embedding_model.encode(chunk)
            
            # Store in database
            cursor.execute(
                "INSERT INTO documents (content, embedding) VALUES (%s, %s)",
                (chunk, embedding.tolist())
            )
        print(f"Successfully stored {len(chunks)} embeddings.")
    else:
        print(f"Found {count} existing embeddings. Skipping embedding generation.")
    
    connection.commit()
    cursor.close()
    connection.close()

# Semantic search function with re-ranking
def semantic_search(query, top_k=10, threshold=0.8, rerank_top_k=3):
    query_embedding = embedding_model.encode(query)
    
    connection = psycopg2.connect(
        host="172.25.170.149",
        port="5432",
        dbname="postgres",
        user="postgres",
        password="steven123"
    )
    cursor = connection.cursor()
    
    # Convert numpy array to list and cast it properly as a vector
    query_embedding_list = query_embedding.tolist()
    
    # Retrieve top-k results with FAISS based on cosine similarity
    cursor.execute("""
        SELECT content, 1 - (embedding <=> %s::vector) as similarity
        FROM documents
        WHERE 1 - (embedding <=> %s::vector) > %s
        ORDER BY similarity DESC
        LIMIT %s
    """, (query_embedding_list, query_embedding_list, threshold, top_k))
    
    results = cursor.fetchall()
    initial_results = [(content, float(similarity)) for content, similarity in results]
    
    cursor.close()
    connection.close()
    
    # Re-ranking using Cross-Encoder for fine-grained semantic scoring
    if initial_results:
        content_texts = [content for content, _ in initial_results]
        scores = cross_encoder.predict([(query, content) for content in content_texts])
        ranked_results = sorted(zip(content_texts, scores), key=lambda x: x[1], reverse=True)
        final_results = ranked_results[:rerank_top_k]
    else:
        final_results = []
    
    return final_results

def generate_llm_response(query, rag_results):
    client = Groq(
        api_key="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"  # Replace with your actual Groq API key
    )
    
    # Construct context from RAG results
    context = "\n".join([content for content, _ in rag_results])
    
    # Construct the prompt
    system_prompt = """You are a helpful AI assistant. Using the provided context, 
    answer the user's question accurately and concisely. If the context doesn't 
    contain relevant information, acknowledge that and provide a general response."""
    
    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": f"""Context: {context}\n\nQuestion: {query}
            
            Please answer the question based on the context provided above."""
        }
    ]
    
    # Get response from Groq
    chat_completion = client.chat.completions.create(
        messages=messages,
        model="mixtral-8x7b-32768",  # or your preferred Groq model
        temperature=0.7,
    )
    
    return chat_completion.choices[0].message.content

def main():
    # Extract text from PDF
    chunks = extract_text_from_pdf('knowledge.pdf')
    
    # Store embeddings (will now skip if already present)
    store_embeddings(chunks)
    
    # Example search
    query = "Tell me about internal combustion engine"
    rag_results = semantic_search(query, top_k=10, threshold=0.6, rerank_top_k=3)
    
    if not rag_results:
        print("No relevant results found.")
    else:
        # Get LLM response using RAG results
        llm_response = generate_llm_response(query, rag_results)
        print("LLM Response:")
        print(llm_response)
        
        # print("\nRAG Results:")
        # for content, similarity in rag_results:
        #     print(f"Re-Ranked Score: {similarity:.4f}")
        #     print(f"Content: {content}\n")

if __name__ == "__main__":
    main()
