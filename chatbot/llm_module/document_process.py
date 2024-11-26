from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
import os
import openai
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

openai.api_key = os.environ.get("MY_OPENAI_API_KEY")

def split_texts(texts, chunk_size=1000, chunk_overlap=200):
    recursive_text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    documents = [Document(page_content=texts)]
    return recursive_text_splitter.split_documents(documents)

def process_documents(INPUT):
    if INPUT.startswith("http"):
        urls = [INPUT]
        docs = [WebBaseLoader(url).load() for url in urls]
        docs_list = [item for sublist in docs for item in sublist]
    else:
        docs_list = str(INPUT)
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000, chunk_overlap=100
    )
    SPLITS = text_splitter.split_documents(docs_list)
    return SPLITS


def summarize_documents(SPLITS):
    summaries = []

    for SPLIT in SPLITS:
        SPLIT = SPLIT.page_content

        try:
            llm = ChatOpenAI(
                model="gpt-4o-mini",
                api_key=openai.api_key,
                max_tokens=500,
                temperature=0.0,
            )
            prompt = ChatPromptTemplate.from_template(
                "너는 대본 작가야, 대본 작성을 위한 자료조사를 위해 문서를 요약해, 문서 : {SPLIT}"
            )
            chain = prompt | llm | StrOutputParser()
            summary = chain.invoke({"SPLIT": SPLIT})
            summaries.append(summary)

        except Exception as e:
            # 오류 처리: 만약 API 호출 중에 문제가 발생하면 오류 메시지 추가
            print(f"Error summarizing document: {e}")
            summaries.append(f"Error summarizing document: {e}")

    return "".join(summaries)


def generate_script(summaries):
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=openai.api_key,
        max_tokens=3000,
        temperature=0.0,
    )
    prompt = ChatPromptTemplate.from_template(
    """
    persona = script writer
    language = only in korean
    use input,
    refer to sample,
    write about time, character, event,
    write only fact
 
    <sample>
    # title : title of script
    # prologue 1 : song, movie, book, show about subject
    - coontent :
    # prologue 2 : explain about subject
    - coontent :
    # prologue 3 : explain about character
    - coontent :
    # exposition 1 : historical background of subject
    - coontent :
    # exposition 2 : history of character
    - coontent :
    # exposition 3 : beginning of event
    - coontent :
    # development 1 : situation, action, static of character
    - coontent :
    # development 2 : influence of event
    - coontent :
    # development 3 : reaction of people
    - coontent :
    # climax 1 : event and effect bigger
    - coontent :
    # climax 2 : dramatic action, conflict
    - coontent :
    # climax 3 : falling Action
    - coontent :
    # denouement : resolution
    - coontent :
    # epilogue : message, remaining
    - coontent :
    </sample>

    <input>
    {summaries}
    </input>

    """
    )
    chain = prompt | llm | StrOutputParser()
    script = chain.invoke({"summaries": summaries})
    return script

def script_maker(INPUT):
    SPLIT = process_documents(INPUT)
    summaries = summarize_documents(SPLIT)
    script = generate_script(summaries)
    return script
