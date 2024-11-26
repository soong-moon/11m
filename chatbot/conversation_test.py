from llm_module.document_process import *
from llm_module.vectorstore_utils import *
from llm_module.chains import *

PATH = './chatbot/db'
script_path = PATH + '/script_db'
script_db = load_vector_store('script_db', script_path)

from operator import itemgetter
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

openai.api_key = os.environ.get("MY_OPENAI_API_KEY")

def chain_maker(script_db):
    script_retriever = script_db.as_retriever(search_type="similarity", search_kwargs={"k": 1})
    prompt = PromptTemplate.from_template(
    """
    persona : story teller
    language : only korean
    tell dramatic story like talking to friend,
    progress chapter by chapter,
    **hide header like '###'**,
    start chapter with interesting question,
    wait user answer
    give reaction to answer,
    do not use same reaction
    
    # script
    {script}

    #Previous Chat History:
    {chat_history}

    #Question: 
    {question} 
    """
    )

    llm = ChatOpenAI(model="gpt-4o-mini", api_key= openai.api_key, temperature=0.3)

    # 단계 8: 체인(Chain) 생성
    chain = (
        {
            "script" : itemgetter("question") | script_retriever,
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

store ={}
chain = chain_maker(script_db)
h_chain = history_chain(chain, store)

while True:
      print("========================")
      query = input("질문을 입력하세요 : ")
      if query.lower() == "exit":
            print("대화를 종료합니다.")
            break
      response = h_chain.invoke(
    # 질문 입력
    {"question": query},
    # 세션 ID 기준으로 대화를 기록합니다.
    config={"configurable": {"session_id": "test11"}},
)
      print(query)
      print("\n답변:")
      print(response)

print(store['test11'])