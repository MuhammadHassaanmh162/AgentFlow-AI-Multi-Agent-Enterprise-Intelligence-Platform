from langchain_community.retrievers import WikipediaRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)
retriever = WikipediaRetriever()

prompt = ChatPromptTemplate.from_template(
"""You are a history assistant for the general public.

Answer using ONLY the provided Wikipedia passages.

User question: {user_question}

Wikipedia passages:
{retrieved_passages}
"""
)

rag_chain = (
    {
        "retrieved_passages": retriever | (lambda docs: "\n\n".join(doc.page_content for doc in docs)),
        "user_question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

question = "What were the key events of the Battle of Thermopylae?"

response = rag_chain.invoke(question)

print(response)


