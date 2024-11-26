import os
import openai
from langchain_core.prompts import PromptTemplate
from operator import itemgetter
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

openai.api_key = os.environ.get("MY_OPENAI_API_KEY")

def chain_maker(rag_db, script_db):
    qustion_retriever = rag_db.as_retriever(search_type="similarity", search_kwargs={"k": 1})
    script_retriever = script_db.as_retriever(search_type="similarity", search_kwargs={"k": 1})
    prompt = PromptTemplate.from_template(
    """
    당신은 스토리 텔러, 
    친구에게 이야기 하듯 반말로 진행하세요,
    같은 답변을 반복하지 마세요,
    한 내용이 완료되면 흥미유발 질문을 하세요
    사용자의 대답에 적절한 리액션을 하세요,

    # 대본
    {script}

    #Previous Chat History:
    {chat_history}

    #Question: 
    {question} 

    #Context: 
    {context} 

    도입 1부터 시작,
    #Answer:너 그 얘기 알아?"""
    )

    llm = ChatOpenAI(model="gpt-4o-mini", api_key= openai.api_key, temperature=0)

    # 단계 8: 체인(Chain) 생성
    chain = (
        {
            "script" : itemgetter("question") | script_retriever,
            "context": itemgetter("question") | qustion_retriever,
            "question": itemgetter("question"),
            "chat_history": itemgetter("chat_history"),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

def history_chain(chain, memory_store : dict):

    def get_session_history(session_ids):
        print(f"[대화 세션ID]: {session_ids}")
        if session_ids not in memory_store:  # 세션 ID가 store에 없는 경우
            # 새로운 ChatMessageHistory 객체를 생성하여 store에 저장
            memory_store[session_ids] = ChatMessageHistory()
        return memory_store[session_ids]  # 해당 세션 ID에 대한 세션 기록 반환


    # 대화를 기록하는 RAG 체인 생성
    rag_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,  # 세션 기록을 가져오는 함수
        input_messages_key="question",  # 사용자의 질문이 템플릿 변수에 들어갈 key
        history_messages_key="chat_history",  # 기록 메시지의 키
    )
    return rag_with_history