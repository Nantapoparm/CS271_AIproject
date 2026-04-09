from langchain.chains import RetrievalQA  
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

llm = ChatGroq(
    groq_api_key="gsk_your_actual_key_here",
    model_name="llama-3.3-70b-versatile", # แนะนำตัวนี้ (แรงและฉลาด)
    temperature=0
)
def setup_retrieval_qa(db):
    retriever = db.as_retriever(search_kwargs={"k": 3})

    prompt_template = """Your name is AgriGenius...
    CONTEXT: {context}
    QUESTION: {input}""" 

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "input"]
    )

    # ใช้ระบบ Chain ใหม่แทน RetrievalQA (ซึ่งถูกยกเลิกไปแล้วใน 0.3.x)
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    chain = create_retrieval_chain(retriever, combine_docs_chain)
    
    return chain