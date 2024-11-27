from flask import Flask, request, session, jsonify, Response
import openai
import os
import json
from secret import app_secret_key, OPENAI_API_KEY
from langchain.memory import ConversationBufferMemory
from langchain import ConversationChain
from langchain_community.chat_models import ChatOpenAI
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import re
import warnings

# FutureWarning 억제
warnings.filterwarnings("ignore", category=FutureWarning)

app = Flask(__name__)
app.secret_key = app_secret_key



##### GPT 설정
llm = ChatOpenAI(temperature=0.0, model="gpt-4o", openai_api_key=OPENAI_API_KEY)



# 요약해보기 사용자별 메모리 초기화
summarization_data_store = {}
##### 요약해보기 모델 (GPT)
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
    if user_id not in summarization_data_store:
        summarization_data_store[user_id] = ConversationBufferMemory()
    memory = summarization_data_store[user_id]

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




# 번역해보기 사용자별 메모리 초기화
translation_data_store = {}
##### 번역해보기 모델 (GPT)
@app.route('/try-translate', methods=['POST'])
def try_translate():
    user_id = request.headers.get('user')
    user_message = request.json.get('message')
    news_content = request.json.get('news_content', None)

    if not user_id:
        return jsonify({'error': 'user_id is required'}), 400
    if not user_message:
        return jsonify({'error': 'message is required'}), 400

    # 사용자별 메모리 객체 생성 또는 가져오기
    if user_id not in translation_data_store:
        translation_data_store[user_id] = ConversationBufferMemory()
    memory = translation_data_store[user_id]

    # 대화 체인 초기화 (사용자별로 메모리와 연결된 대화 체인 생성)
    conversation_chain = ConversationChain(
        llm     = llm,
        memory  = memory,
        verbose = True  # 대화 히스토리 확인용
    )

    # 첫 번째 통신일 경우 뉴스 기사 내용 추가
    if news_content and not memory.chat_memory.messages:
        gpt_response= conversation_chain.predict(input=f"뉴스 문장: {news_content}\n \n 사용자가 뉴스 문장을 한글은 영어로, 영어는 한글로 번역한 문장: {user_message}\n\n사용자가 뉴스 문장을 번역한 문장에 대해 문법적, 내용적 등 다양한 기준을 고려하여 상세히 피드백을 해주세요. 사용자는 영어문장은 한글로 번역하고, 한글은 영어로 번역함으로써 영어 공부를 하고자 합니다. 사용자는 이를 통해 영어 공부를 하고자 하는 것이니 목적에 맞게 답변 해주세요. 뉴스 문장을 제대로 번역했는지 확인하세요! 그리고 당신은 한글로 피드백 해주세요!")
    else:
        # 사용자 메시지 처리
        gpt_response = conversation_chain.predict(input=f"{user_message} \n 이는 이전 번역역습내용에 대한 추가적인 질문 혹은 사용자가다시 작성해본 한글은 영어로, 영어는 한글로 번역한 것입니다. 사용자는 이를 통해 영어 공부를 하고자 하는 것이니 목적에 맞게 답변 해주세요. 그리고 당신은 한글로 피드백 해주세요!")
        
    # 줄바꿈과 UTF-8 인코딩을 유지하여 JSON 형태로 반환
    response_data = {
        'gpt_answer': gpt_response
    }
    return Response(json.dumps(response_data, ensure_ascii=False, indent=2), content_type='application/json; charset=utf-8')




# 영어 문장 뜯어보기 기능 (GPT)
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




# 기사 통요약 기능(한글, 영어 둘다) (GPT)
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
                                    3. 어느정도 길어도 되니 핵심내용은 모두 포함하여 요약해주세요!
                                    ''')

    # 줄바꿈과 UTF-8 인코딩을 유지하여 JSON 형태로 반환
    response_data = {
        'gpt_answer': gpt_response
    }
    return Response(json.dumps(response_data, ensure_ascii=False, indent=2), content_type='application/json; charset=utf-8')




# 기사 통번역 기능 (한<->영) (GPT)
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



############ 학습 모델 사용
# 통요약 모델 by BigBird-Peagsus
@app.route('/summarize_e', methods=['POST'])
def summarize_e():
    auth_token = os.getenv("HUGGINGFACE_TOKEN")

    user_id = request.headers.get('user')
    news_content = request.json.get('news_content')

    if not user_id:
        return jsonify({'errorS': 'user_id is required'}), 400
    if not news_content:
        return jsonify({'error': 'content is required'}), 400
    
    # 추가된 토큰 초기화
    tokenizer = T5Tokenizer.from_pretrained('t5-base', use_auth_token=auth_token)  # 저장된 토크나이저 경로
    # tokenizer = T5Tokenizer.from_pretrained('tokenizer/summarize_tokenizer', use_auth_token=auth_token)  # 저장된 토크나이저 경로
    model = T5ForConditionalGeneration.from_pretrained('models/summarize_model.pth', use_auth_token=auth_token)  # 저장된 모델 경로
    model.eval()
    
    input_text = f"summarize: {news_content}"
    input_ids = tokenizer.encode(input_text, return_tensors="pt", truncation=True, max_length=1024)

    # Generate summary
    summary_ids = model.generate(input_ids, max_length=300, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    # 줄바꿈과 UTF-8 인코딩을 유지하여 JSON 형태로 반환
    response_data = {
        'answer': summary
    }
    return Response(json.dumps(response_data, ensure_ascii=False, indent=2), content_type='application/json; charset=utf-8')




@app.route('/translate_t5_k2e', methods=['POST'])
def translate_t5_k2e():
    user_id = request.headers.get('user')
    news_content = request.json.get('news_content')

    if not user_id:
        return jsonify({'error': 'user_id is required'}), 400
    if not news_content:
        return jsonify({'error': 'content is required'}), 400

    # 모델 및 토크나이저 로드
    model_ckpt = "KETI-AIR/ke-t5-base"
    model_save_path = "models/model_weights_kotoen.pth"
    max_token_length = 256  # 최대 토큰 길이 증가
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_ckpt)
    
    # 가중치 로드
    model.load_state_dict(torch.load(model_save_path,map_location=torch.device('cpu')))
    model.eval()

    sentences = re.split(r'(?<=\.)\s+', news_content)   # 마침표 후에 공백을 기준으로 문장 분리

    def batch_translate(sentences, batch_size=8):
        translated_sentences = []
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i+batch_size]
            
            # 입력 처리 (배치 처리)
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_token_length)
            
            # 모델 출력 생성 (FP16 precision 사용)
            inputs = {key: value for key, value in inputs.items()}  # GPU로 데이터 이동
            outputs = model.generate(**inputs, num_beams=3, max_length=max_token_length, early_stopping=True)
            
            # 배치 번역 결과 디코딩
            batch_translations = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            translated_sentences.extend(batch_translations)
        
        return translated_sentences

    # 배치 단위로 번역
    translated_sentences = batch_translate(sentences, batch_size=16)
    translated_sentences = "".join(translated_sentences)

    # 줄바꿈과 UTF-8 인코딩을 유지하여 JSON 형태로 반환
    response_data = {
        'answer': translated_sentences
    }
    return Response(json.dumps(response_data, ensure_ascii=False, indent=2), content_type='application/json; charset=utf-8')




@app.route('/translate_t5_e2k', methods=['POST'])
def translate_t5_e2k():
    user_id = request.headers.get('user')
    news_content = request.json.get('news_content')

    if not user_id:
        return jsonify({'error': 'user_id is required'}), 400
    if not news_content:
        return jsonify({'error': 'content is required'}), 400

    # 모델 및 토크나이저 로드
    model_ckpt = "KETI-AIR/ke-t5-base"
    model_save_path = "models/model_weights.pth"
    max_token_length = 256  # 최대 토큰 길이 증가
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_ckpt)
    
    # 가중치 로드
    model.load_state_dict(torch.load(model_save_path, map_location=torch.device('cpu')))
    model.eval()

    sentences = re.split(r'(?<=\.)\s+', news_content)   # 마침표 후에 공백을 기준으로 문장 분리

    def batch_translate(sentences, batch_size=8):
        translated_sentences = []
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i+batch_size]
            
            # 입력 처리 (배치 처리)
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_token_length)
            
            # 모델 출력 생성 (FP16 precision 사용)
            inputs = {key: value for key, value in inputs.items()}  # GPU로 데이터 이동
            outputs = model.generate(**inputs, num_beams=3, max_length=max_token_length, early_stopping=True)
            
            # 배치 번역 결과 디코딩
            batch_translations = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            translated_sentences.extend(batch_translations)
            translated_sentences = "".join(translated_sentences)
        return translated_sentences

    # 배치 단위로 번역
    translated_sentences = batch_translate(sentences, batch_size=16)

    # 줄바꿈과 UTF-8 인코딩을 유지하여 JSON 형태로 반환
    response_data = {
        'answer': translated_sentences
    }
    return Response(json.dumps(response_data, ensure_ascii=False, indent=2), content_type='application/json; charset=utf-8')


    
    
if __name__ == "__main__":
    app.run(debug=True)