import warnings
import os
import openai
from openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from operator import itemgetter
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain")
openai.api_key = os.environ.get("NBCAMP_01")


def load_and_process_documents(urls):
    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]

    # 문서 분할:
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000, chunk_overlap=100
    )
    doc_splits = text_splitter.split_documents(docs_list)
    return doc_splits


def summarize_documents(docs):
    summaries = []

    for doc in docs:
        doc_text = doc.page_content

        # 요약을 요청하는 프롬프트 생성
        try:
            client = OpenAI(api_key=openai.api_key)
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",  # 모델 선택
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant.",
                    },  # 시스템 메시지
                    {
                        "role": "user",
                        "content": f"이 문서를 요약해주세요: {doc_text}",
                    },  # 요약 요청
                ],
                max_tokens=200,  # 요약된 내용의 최대 토큰 길이
                temperature=0.5,  # 생성의 창의성 조정 (0은 정해진 답변, 1은 더 창의적)
            )

            # 요약된 텍스트 추출
            summary = response.choices[
                0
            ].message.content  # 응답에서 요약된 텍스트 가져오기
            summaries.append(summary)  # 요약된 문서 리스트에 추가

        except Exception as e:
            # 오류 처리: 만약 API 호출 중에 문제가 발생하면 오류 메시지 추가
            print(f"Error summarizing document: {e}")
            summaries.append(f"Error summarizing document: {e}")

    return "".join(summaries)


def generate_script_from_summary(summarized_text):
    # 대본 작성을 위한 템플릿 생성
    script_prompt = f"""
    persona = 대본 작가
    language = 한국어로만 답합니다.
    
    <rule>
    개조식으로 작성
    회차는 10회
    회차는 시간 순서대로 진행
    </rule>

    <sample>
    대본 제목 : 
    회차 :
    제목 :
    배경 :
    사건 :
    인물 :
    중요 장면 :
    </sample>

    <output>
    1회부터 10회까지의 대본
    </output>
    """

    # OpenAI를 통해 대본 생성
    client = OpenAI(api_key=openai.api_key)
    script = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": f"{script_prompt}"},  # 시스템 메시지
            {"role": "user", "content": f"{summarized_text}"},
        ],
        max_tokens=3000,
    )
    return script.choices[0].message.content


def create_vector_store(DOCS, db_name: str):
    return Chroma.from_documents(
        documents=DOCS,
        collection_name=db_name,
        embedding=OpenAIEmbeddings(
            model="text-embedding-ada-002", api_key=openai.api_key
        ),
    )


urls = ["https://namu.wiki/w/77246%20위조지폐%20유통사건"]
docs = load_and_process_documents(urls)
summary = summarize_documents(docs)
script = generate_script_from_summary(summary)
script_documents = [
    Document(page_content=script, metadata={"source": "script"}, id="1"),
]
vdb = create_vector_store(docs, "vdb")
# vdb.add_documents(script_documents)
sdb = create_vector_store(script_documents, "sdb")
qustion_retriever = vdb.as_retriever(search_type="similarity", search_kwargs={"k": 1})
script_retriever = sdb.as_retriever(search_type="similarity", search_kwargs={"k": 1})


def script_finder(querry):
    result = script_retriever.invoke(querry)
    prompt = f"""
    persona : you are script search system
    find script about querry
    <querry>
    {querry}
    </querry>

    calculate relavence score
    between querry and script
    socre = [1-100]

    <script>
    {script}
    </script>

    return only integer
    """

    client = OpenAI(api_key=openai.api_key)
    score = client.chat.completions.create(
        model="gpt-3.5-turbo",
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
        return [int(score.choices[0].message.content), result[0]]


while True:
    print("================================")
    querry = input("어떤 이야기가 듣고 싶으신가요?")
    if querry.lower() == "exit":
        break
    relavence = script_finder(querry)
    if relavence[0] < 80:
        print("모르는 이야기 입니다.")
        user_input = input("이야기를 생성하려면 텍스트 또는 URL을 입력하세요: ")
        if user_input.startswith("http"):  # URL이면 문서 로드
            url = [user_input]
            docs = load_and_process_documents(url)
            summary = summarize_documents(docs)
            script = generate_script_from_summary(summary)
            script_documents = [
                Document(page_content=script, metadata={"source": "script"}),
            ]
            sdb.add_documents(script_documents)
            print("이야기가 생성되었습니다.")
            continue
        else:
            context = str(user_input)
            summary = summarize_documents(docs)
            script = generate_script_from_summary(summary)
            script_documents = [
                Document(page_content=script, metadata={"source": "script"}),
            ]
            sdb.add_documents(script_documents)
            print("이야기가 생성되었습니다.")
            continue
    elif relavence[0] >= 80 and relavence[0] < 90:
        print("더 자세히 이야기 해주세요")
        continue
    elif relavence[0] >= 95:
        script = relavence[1]
        print(script)
        break

while True:
      print("========================")
      query = input("질문을 입력하세요 : ")
      if query.lower() == "exit":
            print("대화를 종료합니다.")
            break
      response = rag_with_history.invoke(
    # 질문 입력
    {"question": query},
    # 세션 ID 기준으로 대화를 기록합니다.
    config={"configurable": {"session_id": "rag123"}},
)
      print(query)
      print("\n답변:")
      print(response)  
