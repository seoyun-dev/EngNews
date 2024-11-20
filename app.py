from flask import Flask, request, session, jsonify, Response
import openai
import json
from secret import app_secret_key, OPENAI_API_KEY
from langchain.memory import ConversationBufferMemory
from langchain import ConversationChain
from langchain_community.chat_models import ChatOpenAI

app = Flask(__name__)
app.secret_key = app_secret_key



##### GPT 설정
llm = ChatOpenAI(temperature=0.0, model="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)



##### 요약해보기 모델 (GPT)
# 사용자별 메모리 초기화
data_store = {}
# app
@app.route('/try-summarize', methods=['POST'])
def try_summarize():
    user_id = request.headers.get('user')
    user_message = request.json.get('message')
    news_content = request.json.get('news_content', None)

    if not user_id:
        return jsonify({'error': 'user_id is required'}), 400
    if not user_message:
        return jsonify({'error': 'message is required'}), 400

    # 사용자별 메모리 객체 생성 또는 가져오기
    if user_id not in data_store:
        data_store[user_id] = ConversationBufferMemory()
    memory = data_store[user_id]

    # 대화 체인 초기화 (사용자별로 메모리와 연결된 대화 체인 생성)
    conversation_chain = ConversationChain(
        llm     = llm,
        memory  = memory,
        verbose = True  # 대화 히스토리 확인용
    )

    # 첫 번째 통신일 경우 뉴스 기사 내용 추가
    if news_content and not memory.chat_memory.messages:
        gpt_response= conversation_chain.predict(input=f"News: {news_content}\n위는 요약하고자 하는 뉴스내용입니다. \n User Summary: {user_message}\n아래는 사용자가 위의 뉴스를 영어로 요약한 내용입니다.\n\n이 요약본에 대해 문법적, 내용적 측면 뿐만 아니라 논리적 흐름, 일관성, 핵심 정보 강조, 명확성, 객관성, 독자의 이해 가능성, 어휘의 적절성, 그리고 중요한 내용 누락 여부를 고려하여 상세히 피드백을 해주세요. 사용자는 이를 통해 영어 공부를 하고자 하는 것이니 목적에 맞게 답변 해주세요. 그리고 당신은 한글로 대답해주세요!")
    else:
        # 사용자 메시지 처리
        gpt_response = conversation_chain.predict(input=f"{user_message} \n 이는 이전 대화내용에 대한 추가적인 질문 혹은 다시 작성해본 요약본입니다. 사용자는 이를 통해 영어 공부를 하고자 하는 것이니 목적에 맞게 답변 해주세요. 그리고 당신은 한글로!!! 대답해주세요!")
        
    # 줄바꿈과 UTF-8 인코딩을 유지하여 JSON 형태로 반환
    response_data = {
        'gpt_answer': gpt_response
    }
    return Response(json.dumps(response_data, ensure_ascii=False, indent=2), content_type='application/json; charset=utf-8')



##### 번역해보기 모델 (GPT)
@app.route('/try-translate', methods=['POST'])
def try_translate():
    return



##### 요약 모델 
@app.route('/summarize', methods=['POST'])
def summarize():
    return 



if __name__ == "__main__":
    app.run(debug=True)