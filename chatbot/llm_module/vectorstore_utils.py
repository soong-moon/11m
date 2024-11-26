from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
import os
import openai
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings

openai.api_key = os.environ.get("MY_OPENAI_API_KEY")


def create_vector_store(DOCS, db_name: str, DB_PATH):
    return Chroma.from_documents(
        documents=DOCS,
        collection_name=db_name,
        persist_directory=DB_PATH,
        embedding=OpenAIEmbeddings(
            model="text-embedding-ada-002", api_key=openai.api_key
        ),
    )

def load_vector_store(db_name: str, DB_PATH):
        return Chroma(
        collection_name=db_name,
        persist_directory=DB_PATH,
        embedding_function=OpenAIEmbeddings(
            model="text-embedding-ada-002", api_key=openai.api_key
        ),
    )


def script_finder(querry, db):
    script_retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 1})
    script = script_retriever.invoke(querry)
    prompt = f"""
    persona : you are script search system
    find script about querry
    <querry>
    {querry}
    </querry>

    calculate relavence score
    between querry and script
    socre = [1-100]
    fail = 0

    <script>
    {script}
    </script>

    return only score
    """

    client = OpenAI(api_key=openai.api_key)
    score = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": f"{prompt}"},
            {"role": "user", "content": f"{querry}"},
        ],
    )
    if int(score.choices[0].message.content) < 80:
        return [int(score.choices[0].message.content)]
    elif (
        int(score.choices[0].message.content) >= 90
        and int(score.choices[0].message.content) < 95
    ):
        return [int(score.choices[0].message.content)]
    elif int(score.choices[0].message.content) >= 90:
        return [int(score.choices[0].message.content), script[0]]
