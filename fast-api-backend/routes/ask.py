from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from models.query_request import QueryRequest
from core.llm import get_llm
from core.embeddings import vector_search
from langchain.schema import HumanMessage
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

router = APIRouter()
llm = get_llm()


# Simple user memory store
user_memories = {}

# @router.post("/ask")
# async def ask(request: QueryRequest, user_id: str):
#     query_text = request.text
#     metadata = request.metadata
    
#     # Get search results
#     top_docs = vector_search.search_by_metadata(metadata, query_text, k=10)
    
#     # Build context from documents
#     context_text = "\n\n".join([
#         f"[File: {doc.get('metadata', {}).get('filename', 'Unknown')}]\n{doc.get('content', '')[:500]}" 
#         for doc in top_docs
#     ])
    
#     # Build policy info
#     policy_info = f"""Policy: {metadata.get('policy_name', 'N/A')}
# Base File: {metadata.get('metadata', {}).get('base_policy_filename', 'N/A')}
# Endorsements: {', '.join([v for k,v in metadata.get('metadata', {}).items() if k.startswith('endorsement')])}"""
    
#     # Initialize user memory if not exists  
#     if user_id not in user_memories:
#         user_memories[user_id] = ConversationBufferWindowMemory(
#             k=10,  # Keep last 10 exchanges
#             return_messages=False,
#             memory_key="chat_history",
#             input_key="input"
#         )
    
#     memory = user_memories[user_id]
    
#     # Create prompt with memory placeholder
#     prompt = PromptTemplate(
#         input_variables=["chat_history", "input"],
#         template="""You are an expert insurance assistant. Answer based on documents and conversation history.

# {policy_info}

# Documents:
# {context_text}

# Previous conversation:
# {chat_history}

# User Question: {input}
# Assistant:"""
#     )
    
#     # Create chain
#     chain = LLMChain(
#         llm=llm,
#         prompt=prompt,
#         memory=memory,
#         verbose=False
#     )
    
#     try:
#         # Get response - memory automatically adds chat_history
#         response = await chain.acall({
#             "input": query_text,
#             "policy_info": policy_info,
#             "context_text": context_text
#         })
#         print(f"Response {response}")
        
#         return {
#             "query": query_text,
#             "gemini_answer": response["text"]
#         }
        
#     except Exception as e:
#         return {
#             "query": query_text,
#             "gemini_answer": f"Error: {str(e)}",
#             "error": True
#         }


@router.post("/ask")
async def ask(request: QueryRequest, user_id: str):
    query_text = request.text
    metadata = request.metadata

    # Get search results
    top_docs = vector_search.search_by_metadata(metadata, query_text, k=20)

    # Build context from documents
    context_text = "\n\n".join([
        f"[File: {doc.get('metadata', {}).get('filename', 'Unknown')}]\n{doc.get('content', '')}"
        for doc in top_docs
    ])

    # Build policy info using metadata
    base_file = metadata.get("base_policy_filename", "N/A")
    endorsements = [
        v for k, v in metadata.items() if k.startswith("endorsement")
    ]
    endorsement_list = "\n".join([f"- {e}" for e in endorsements]) if endorsements else "None"

    policy_info = f"""
Policy Information:
- Base File: {base_file}
- Endorsements:
{endorsement_list}
"""

    # Initialize user memory if not exists
    if user_id not in user_memories:
        user_memories[user_id] = ConversationBufferWindowMemory(
            k=10,  # Keep last ~2000 exchanges
            return_messages=False,
            memory_key="chat_history",
            input_key="input"
        )

    memory = user_memories[user_id]

    # Prompt
    prompt = PromptTemplate(
        input_variables=["chat_history", "input", "policy_info", "context_text"],
        template=f"""You are an expert insurance assistant specializing in policy interpretation. 
Your role is to help the user understand their insurance policy, using the provided documents and metadata. 
Never invent information outside of the given documents.

### Policy Metadata
{{policy_info}}

- The **base policy file** is the primary source of truth.  
- **Endorsements** may modify, replace, or override sections of the base policy.  
- Always reconcile conflicts by prioritizing endorsements over the base policy.  
- If multiple endorsements exist, assume later ones override earlier ones.  

### Available Documents
{{context_text}}

### Instructions
1. Use only the information provided in the above documents and metadata.  
2. When referencing content, mention the filename (e.g., "According to endorsement1.pdf").  
3. If the answer is unclear or not covered by the documents, explicitly say:  
   *"This information is not provided in the available policy documents."*  
4. Maintain clarity, structure, and accuracy in your explanations.  
5. Keep answers concise but thorough.  

### Previous Conversation
{{chat_history}}

### User Question
{{input}}

### Assistant Response:"""
    )

    # Create chain
    chain = LLMChain(
        llm=llm,
        prompt=prompt,
        memory=memory,
        verbose=False
    )

    try:
        # Get response - memory automatically adds chat_history
        response = await chain.acall({
            "input": query_text,
            "policy_info": policy_info,
            "context_text": context_text
        })
        # print(f"Response {response}")

        return {
            "query": query_text,
            "gemini_answer": response["text"],
            "top_Docs":top_docs
        }

    except Exception as e:
        return {
            "query": query_text,
            "gemini_answer": f"Error: {str(e)}",
            "error": True
        }




@router.post("/ask/stream")
async def ask_stream(request: QueryRequest):
    query_text = request.text

    system_prompt = f"""
You are an expert insurance policy assistant. Use the PDF contents to answer.
User Question: {query_text}
"""

    async def token_generator():
        try:
            async for chunk in llm.astream([HumanMessage(content=system_prompt)]):
                text = getattr(chunk, "text", None)
                print("Text",text)
                if not text or callable(text):
                    continue
                # Don't add extra newlines - send clean text
                yield text.encode("utf-8")
        except Exception as e:
            print(f"Streaming error: {e}")
            yield f"Error: {str(e)}".encode("utf-8")

    return StreamingResponse(
        token_generator(), 
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )