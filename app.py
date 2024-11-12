# server.py
from flask import Flask, request, json, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# 학습된 머신러닝 모델 로드
# model = joblib.load('summarizer_model.pkl')  # 모델을 사전에 저장해둔다고 가정

@app.route('/summarize', methods=['GET'])
def summarize():
    return "hello"
    try:
        # 요청에서 JSON 데이터를 받기
        data = request.get_json()
        news_content = data.get('content')

        # 모델을 이용해 요약 생성 (여기서는 간단히 길이 줄이기로 예시)
        summary = model.summarize(news_content)  # 가정: 모델의 summarize 메소드가 요약을 수행

        # 결과를 JSON 형식으로 반환
        return jsonify({'summary': summary}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500



@app.route('/try-summarizing', methods=['GET'])
def try_summarizing():
    return "try"
    try:
        # 요청에서 JSON 데이터를 받기
        data = request.get_json()
        news_content = data.get('content')

        # 모델을 이용해 요약 생성 (여기서는 간단히 길이 줄이기로 예시)
        summary = model.try_summarizing(news_content)  # 가정: 모델의 summarize 메소드가 요약을 수행

        # 결과를 JSON 형식으로 반환
        return jsonify({'summary': summary}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000)