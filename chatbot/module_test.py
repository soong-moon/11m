from llm_module.document_process import *
from llm_module.vectorstore_utils import *
from llm_module.chains import *

PATH = './chatbot/db'
rag_path = PATH + '/rag_db'
script_path = PATH + '/script_db'
url = "https://namu.wiki/w/77246%20위조지폐%20유통사건"
docs = process_documents(url)
script = script_maker(url)
script_documents = [
                Document(page_content=script, metadata={"source": "script"}),
            ]
vdb = create_vector_store(docs, 'rag_db', rag_path)
sdb = create_vector_store(script_documents, 'script_db', script_path)

# while True:
#     print("================================")
#     querry = input("어떤 이야기가 듣고 싶으신가요?")
#     if querry.lower() == "exit":
#         break
#     relavence = script_finder(querry, sdb)
#     if relavence[0] < 80:
#         print("모르는 이야기 입니다.")
#         user_input = input("이야기를 생성하려면 텍스트 또는 URL을 입력하세요: ")
#         new_script = script_maker(user_input)
#         script_documents = [
#                 Document(page_content=new_script, metadata={"source": "script"}),
#             ]
#         sdb.add_documents(script_documents)
#         print("이야기가 생성되었습니다.")
#         continue
#     elif relavence[0] >= 80 and relavence[0] < 90:
#         print("더 자세히 이야기 해주세요")
#         continue
#     elif relavence[0] >= 95:
#         script = relavence[1]
#         print(script)
#         break

# store ={}
# chain = chain_maker(vdb, sdb)
# h_chain = history_chain(chain, store)
# response = h_chain.invoke(
#     # 질문 입력
#     {"question": querry},
#     # 세션 ID 기준으로 대화를 기록합니다.
#     config={"configurable": {"session_id": "test6"}},
# )

# while True:
#       print("========================")
#       query = input("질문을 입력하세요 : ")
#       if query.lower() == "exit":
#             print("대화를 종료합니다.")
#             break
#       response = h_chain.invoke(
#     # 질문 입력
#     {"question": query},
#     # 세션 ID 기준으로 대화를 기록합니다.
#     config={"configurable": {"session_id": "test6"}},
# )
#       print(query)
#       print("\n답변:")
#       print(response)  