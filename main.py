from utilities.knowledge_extractor import extract_text_from_pdf
from utilities.db_operations import store_embeddings, semantic_search
from utilities.llm_call import generate_llm_response

def main():
    chunks = extract_text_from_pdf('utilities/knowledge.pdf')
    
    store_embeddings(chunks)
    
    query = "Tell me about internal combustion engine"
    rag_results = semantic_search(query, top_k=10, threshold=0.6, rerank_top_k=3)
    
    if not rag_results:
        print("No relevant results found.")
    else:
        llm_response = generate_llm_response(query, rag_results)
        print("LLM Response:")
        print(llm_response)
        print("\nRAG Results:")
        print(rag_results)

if __name__ == "__main__":
    main()
