from flask import Flask, request

app = Flask(__name__)

@app.route('/')
def home():
    return "Hello, Flask!"

@app.route('/greet', methods=['GET'])
def greet():
    name = request.args.get('name', 'Guest')  # URL 파라미터로 'name' 받기, 없으면 기본값은 'Guest'
    return f"Hello, {name}!"

if __name__ == "__main__":
    app.run(debug=True)