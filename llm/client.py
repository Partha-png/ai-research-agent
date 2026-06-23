import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()


def get_llm():

    return ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.2
    )


def invoke_llm(prompt: str):

    llm = get_llm()

    response = llm.invoke(prompt)

    return response.content