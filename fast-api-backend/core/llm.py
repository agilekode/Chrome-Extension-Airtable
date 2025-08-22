from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
import os

def get_llm():
    # return ChatGoogleGenerativeAI(
    #     model="gemini-2.5-flash-lite",
    #     google_api_key=os.getenv("GOOGLE_API_KEY"),
    #     temperature=0.1,
    #     streaming=True
        
    # )
    return ChatOpenAI(
        model="gpt-5-nano",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0.1,   # very deterministic
    )
