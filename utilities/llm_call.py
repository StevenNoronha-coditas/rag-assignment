from utilities.pydantic_files import LLMResponse
from groq import Groq

def generate_llm_response(query, rag_results):
    client = Groq()
    
    context = "\n".join([result.content for result in rag_results])
    
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
    
    chat_completion = client.chat.completions.create(
        messages=messages,
        model="mixtral-8x7b-32768",
        temperature=0.7,
    )
    
    llm_response_text = chat_completion.choices[0].message.content
    return LLMResponse(context=context, response=llm_response_text)