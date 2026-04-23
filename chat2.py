from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

llm = ChatGroq(
    groq_api_key="gsk_xxxxxxxxxxLOCALHOSTXXXXXXXXXXXX",
    model_name="llama-3.3-70b-versatile",
    temperature=0
)

def setup_retrieval_qa(db):
    retriever = db.as_retriever(search_kwargs={"k": 3})

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Your name is AgriGenius. Use the context to answer.\n\nCONTEXT: {context}"),
        ("human", "{input}")
    ])

    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    chain = create_retrieval_chain(retriever, combine_docs_chain)
    
    return chain
