from flask import Flask, request, session, jsonify, Response
import requests
import os
import json
from secret import app_secret_key, OPENAI_API_KEY
from langchain.memory import ConversationBufferMemory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
import re
import warnings

# # FutureWarning 억제
# warnings.filterwarnings("ignore", category=FutureWarning)

app = Flask(__name__)
app.secret_key = app_secret_key

# ##### GPT 설정
llm = ChatOpenAI(temperature=0.0, model="gpt-4o", openai_api_key=OPENAI_API_KEY)
# # 사용자별 메모리 초기화
data_store = {}

# 사용자별 메모리 객체 생성 또는 가져오기
def get_session_history(user_id: str) -> InMemoryChatMessageHistory:
    if user_id not in data_store:
        data_store[user_id] = InMemoryChatMessageHistory()
    return data_store[user_id]

##### 요약해보기 모델 (GPT)
@app.route('/try-summarize', methods=['POST'])
def try_summarize():
    user_id = request.headers.get('user')
    news_content = request.json.get('news_content', None)
    user_message = request.json.get('message', None)

    if not user_id:
        return jsonify({'error': 'user_id is required'}), 400

    # 대화 체인 초기화 (사용자별로 메모리와 연결된 대화 체인 생성)
    summarization_chain = RunnableWithMessageHistory(llm, get_session_history)

    # 사용자별 데이터 초기화 확인
    if user_id not in data_store:
        data_store[user_id] = InMemoryChatMessageHistory()

    # 사용자별 이전 뉴스 문장을 저장하고 가져오기
    old_news_sentence = data_store.get(user_id, None)

    # news_sentence가 변경되면 메모리 초기화 (news_sentence가 None이 아닐 때만)
    if news_content is not None and (old_news_sentence != news_content):
        data_store[user_id].clear()  # 대화 체인 메모리 초기화
        data_store[user_id] = InMemoryChatMessageHistory()
        gpt_response = summarization_chain.invoke(
            f"NEWS CONTENT : {news_content} \n 사용자는 위 내용을 한글 또는 영어로 요약하며 영어 공부를 할거야. 사용자가 요약하도록 \"안녕하세요! 뉴스 내용을 요약해보세요!\" 라고만 말해줘!",
            config={"configurable": {"session_id": user_id}}
        )
    else:
        gpt_response = summarization_chain.invoke(
            f'''사용자의 질문 or 요약본: {user_message} \n
            사용자는 gpt와의 대화를 통해 요약 공부를 하고자 합니다. 당신은 친절한 영어강사가 되어 사용자의 질문or요약본에 대답해주세요. 그리고 사용자는 한국사람이므로 한글로 피드백 해주세요!''',
            config={"configurable": {"session_id": user_id}}
        )
        
    # gpt_response에서 필요한 내용 추출
    response_content = gpt_response.content if hasattr(gpt_response, 'content') else str(gpt_response)
    
    # 줄바꿈과 UTF-8 인코딩을 유지하여 JSON 형태로 반환
    response_data = {
        'gpt_answer': response_content
        # 'data_store' : str(data_store[user_id])
    }
    return Response(json.dumps(response_data, ensure_ascii=False, indent=2), content_type='application/json; charset=utf-8')



##### 번역해보기 모델 (GPT)
@app.route('/try-translate', methods=['POST'])
def try_translate():
    user_id = request.headers.get('user')
    news_sentence = request.json.get('news_sentence', None)
    user_message = request.json.get('message', None)

    if not user_id:
        return jsonify({'error': 'user_id is required'}), 400

    # 대화 체인 초기화 (사용자별로 메모리와 연결된 대화 체인 생성)
    translation_chain = RunnableWithMessageHistory(llm, get_session_history)

    # 사용자별 데이터 초기화 확인
    if user_id not in data_store:
        data_store[user_id] = InMemoryChatMessageHistory()

    # 사용자별 이전 뉴스 문장을 저장하고 가져오기
    old_news_sentence = data_store.get(user_id, None)

    # news_sentence가 변경되면 메모리 초기화 (news_sentence가 None이 아닐 때만)
    if news_sentence is not None and (old_news_sentence != news_sentence):
        data_store[user_id].clear()  # 대화 체인 메모리 초기화
        data_store[user_id] = InMemoryChatMessageHistory()
        gpt_response = translation_chain.invoke(
            f"NEWS CONTENT : {news_sentence} \n 사용자는 위 내용이 한글이면 영어로, 영어면 한글로 번역하며 영어 공부를 할거야. 사용자가 번역하도록 \"안녕하세요! 뉴스 문장을 번역해보세요!\" 라고만 말해줘!",
            config={"configurable": {"session_id": user_id}}
        )
    else:
        gpt_response = translation_chain.invoke(
            f'''사용자의 질문 or 번역본: {user_message} \n
            사용자는 gpt와의 대화를 통해 한글 영어 번역 공부를 하고자 합니다. 당신은 친절한 영어강사가 되어 사용자의 질문or번역본에 대답해주세요. 그리고 사용자는 한국사람이므로 한글로 피드백 해주세요!''',
            config={"configurable": {"session_id": user_id}}
        )
        
    # gpt_response에서 필요한 내용 추출
    response_content = gpt_response.content if hasattr(gpt_response, 'content') else str(gpt_response)
    
    # 줄바꿈과 UTF-8 인코딩을 유지하여 JSON 형태로 반환
    response_data = {
        'gpt_answer': response_content
        # 'data_store' : str(data_store[user_id])
    }
    return Response(json.dumps(response_data, ensure_ascii=False, indent=2), content_type='application/json; charset=utf-8')


##### 영어 문장 뜯어보기 기능 (GPT)
@app.route('/analyze-sentence', methods=['POST'])
def analyze_sentence():
    user_id = request.headers.get('user')
    news_sentence = request.json.get('news_sentence')

    if not user_id:
        return jsonify({'error': 'user_id is required'}), 400
    if not news_sentence:
        return jsonify({'error': 'sentence is required'}), 400

    # 중요 단어 및 문장의 숙어 분석
    gpt_response = llm.predict(text=f'''아래 영어 문장을 분석해주세요:\n\n
                                    Sentence: {news_sentence}\n
                                    1. 중요 단어의 뜻과 품사\n 
                                    2. 문장에서 사용된 중요 숙어와 의미\n 
                                    3. 문법적 요소 분석\n
                                    4. 사용자가 영어 공부를 위해 유용하게 참고할 만한 사항들을 포함해주세요.\n
                                    답변은 한글로 해주세요.''')

    # 줄바꿈과 UTF-8 인코딩을 유지하여 JSON 형태로 반환
    response_data = {
        'gpt_answer': gpt_response
    }
    return Response(json.dumps(response_data, ensure_ascii=False, indent=2), content_type='application/json; charset=utf-8')


##### 기사 통요약 기능(한글, 영어 둘다) (GPT)
@app.route('/summarize', methods=['POST'])
def summarize():
    user_id = request.headers.get('user')
    news_content = request.json.get('news_content')

    if not user_id:
        return jsonify({'error': 'user_id is required'}), 400
    if not news_content:
        return jsonify({'error': 'sentence is required'}), 400

    gpt_response = llm.predict(text=f'''아래 News 내용을 한글과 영어로 각각 요약해주세요!:\n\n
                                    News: {news_content}\n
                                    1. 한글: 한글로요약한문장 (한줄띄고) 영어: 영어로 요약한 문장 식으로 보기 쉽게 출력해주세요\n 
                                    2. 문맥이 매끄럽고 이해하기 쉽게 요약해주세요 \n
                                    3. 어느정도 길어도 되니 핵심내용은 모두 포함하여 요약해주세요!''')

    # 줄바꿈과 UTF-8 인코딩을 유지하여 JSON 형태로 반환
    response_data = {
        'gpt_answer': gpt_response
    }
    return Response(json.dumps(response_data, ensure_ascii=False, indent=2), content_type='application/json; charset=utf-8')


##### 기사 통번역 기능 (한<->영) (GPT)
@app.route('/translate', methods=['POST'])
def translate():
    user_id = request.headers.get('user')
    news_content = request.json.get('news_content')

    if not user_id:
        return jsonify({'error': 'user_id is required'}), 400
    if not news_content:
        return jsonify({'error': 'content is required'}), 400

    gpt_response = llm.predict(text=f'''
                                아래 News 내용이 한글이라면 영어로 번역하고, 뉴스내용이 영어라면 한글로 번역해주세요!:\n\n
                                News: {news_content}\n
                                문맥이 매끄럽고 이해하기 쉽게 번역해주세요.''')

    # 줄바꿈과 UTF-8 인코딩을 유지하여 JSON 형태로 반환
    response_data = {
        'gpt_answer': gpt_response
    }
    return Response(json.dumps(response_data, ensure_ascii=False, indent=2), content_type='application/json; charset=utf-8')



# ############ 학습 모델 사용

# # 통요약 모델 by BigBird-Peagsus
# @app.route('/summarize_e', methods=['POST'])
# def summarize_e():
#     auth_token = os.getenv("HUGGINGFACE_TOKEN")

#     user_id = request.headers.get('user')
#     news_content = request.json.get('news_content')

#     if not user_id:
#         return jsonify({'errorS': 'user_id is required'}), 400
#     if not news_content:
#         return jsonify({'error': 'content is required'}), 400
    
#     # 추가된 토큰 초기화
#     tokenizer = T5Tokenizer.from_pretrained('t5-base', use_auth_token=auth_token)  # 저장된 토크나이저 경로
#     # tokenizer = T5Tokenizer.from_pretrained('tokenizer/summarize_tokenizer', use_auth_token=auth_token)  # 저장된 토크나이저 경로
#     model = T5ForConditionalGeneration.from_pretrained('models/summarize_model.pth', use_auth_token=auth_token)  # 저장된 모델 경로
#     model.eval()
    
#     input_text = f"summarize: {news_content}"
#     input_ids = tokenizer.encode(input_text, return_tensors="pt", truncation=True, max_length=1024)

#     # Generate summary
#     summary_ids = model.generate(input_ids, max_length=300, num_beams=4, early_stopping=True)
#     summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

#     # 줄바꿈과 UTF-8 인코딩을 유지하여 JSON 형태로 반환
#     response_data = {
#         'answer': summary
#     }
#     return Response(json.dumps(response_data, ensure_ascii=False, indent=2), content_type='application/json; charset=utf-8')












# # 한>영 통번역 모델 by T5
# @app.route('/translate_t5_k2e', methods=['POST'])
# def translate_t5_k2e():
#     user_id = request.headers.get('user')
#     news_content = request.json.get('news_content')

#     if not user_id:
#         return jsonify({'error': 'user_id is required'}), 400
#     if not news_content:
#         return jsonify({'error': 'content is required'}), 400

#     # 모델 및 토크나이저 로드
#     model_ckpt = "KETI-AIR/ke-t5-base"
#     model_save_path = "models/model_weights_kotoen.pth"
#     max_token_length = 256  # 최대 토큰 길이 증가
#     tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
#     model = AutoModelForSeq2SeqLM.from_pretrained(model_ckpt)
    
#     # 가중치 로드
#     model.load_state_dict(torch.load(model_save_path,map_location=torch.device('cpu')))
#     model.eval()

#     sentences = re.split(r'(?<=\.)\s+', news_content)   # 마침표 후에 공백을 기준으로 문장 분리

#     def batch_translate(sentences, batch_size=8):
#         translated_sentences = []
#         for i in range(0, len(sentences), batch_size):
#             batch = sentences[i:i+batch_size]
            
#             # 입력 처리 (배치 처리)
#             inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_token_length)
            
#             # 모델 출력 생성 (FP16 precision 사용)
#             inputs = {key: value for key, value in inputs.items()}  # GPU로 데이터 이동
#             outputs = model.generate(**inputs, num_beams=3, max_length=max_token_length, early_stopping=True)
            
#             # 배치 번역 결과 디코딩
#             batch_translations = tokenizer.batch_decode(outputs, skip_special_tokens=True)
#             translated_sentences.extend(batch_translations)
        
#         return translated_sentences

#     # 배치 단위로 번역
#     translated_sentences = batch_translate(sentences, batch_size=16)
#     translated_sentences = "".join(translated_sentences)

#     # 줄바꿈과 UTF-8 인코딩을 유지하여 JSON 형태로 반환
#     response_data = {
#         'answer': translated_sentences
#     }
#     return Response(json.dumps(response_data, ensure_ascii=False, indent=2), content_type='application/json; charset=utf-8')




# # 영>한 통번역 모델 by T5
# @app.route('/translate_t5_e2k', methods=['POST'])
# def translate_t5_e2k():
#     user_id = request.headers.get('user')
#     news_content = request.json.get('news_content')

#     if not user_id:
#         return jsonify({'error': 'user_id is required'}), 400
#     if not news_content:
#         return jsonify({'error': 'content is required'}), 400

#     # 모델 및 토크나이저 로드
#     model_ckpt = "KETI-AIR/ke-t5-base"
#     model_save_path = "models/model_weights.pth"
#     max_token_length = 256  # 최대 토큰 길이 증가
#     tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
#     model = AutoModelForSeq2SeqLM.from_pretrained(model_ckpt)
    
#     # 가중치 로드
#     model.load_state_dict(torch.load(model_save_path, map_location=torch.device('cpu')))
#     model.eval()

#     sentences = re.split(r'(?<=\.)\s+', news_content)   # 마침표 후에 공백을 기준으로 문장 분리

#     def batch_translate(sentences, batch_size=8):
#         translated_sentences = []
#         for i in range(0, len(sentences), batch_size):
#             batch = sentences[i:i+batch_size]
            
#             # 입력 처리 (배치 처리)
#             inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_token_length)
            
#             # 모델 출력 생성 (FP16 precision 사용)
#             inputs = {key: value for key, value in inputs.items()}  # GPU로 데이터 이동
#             outputs = model.generate(**inputs, num_beams=3, max_length=max_token_length, early_stopping=True)
            
#             # 배치 번역 결과 디코딩
#             batch_translations = tokenizer.batch_decode(outputs, skip_special_tokens=True)
#             translated_sentences.extend(batch_translations)
#             translated_sentences = "".join(translated_sentences)
#         return translated_sentences

#     # 배치 단위로 번역
#     translated_sentences = batch_translate(sentences, batch_size=16)

#     # 줄바꿈과 UTF-8 인코딩을 유지하여 JSON 형태로 반환
#     response_data = {
#         'answer': translated_sentences
#     }
#     return Response(json.dumps(response_data, ensure_ascii=False, indent=2), content_type='application/json; charset=utf-8')


    
    
if __name__ == "__main__":
    app.run(debug=True)