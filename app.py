import time
from flask import Flask, jsonify, request
from redis import Redis
from rq import Queue
from task import process_question
from random import randint
from flask_caching import Cache
import os
from dotenv import load_dotenv
from celery.result import AsyncResult

load_dotenv()

app = Flask(__name__)
app.config['CACHE_TYPE'] = 'simple'
cache = Cache(app)

# Connect to Redis for task queue management
redis_host = os.getenv('REDIS_HOST', 'localhost')
redis_port = int(os.getenv('REDIS_PORT', 6379))
redis_conn = Redis(host=redis_host, port=redis_port)
q = Queue(connection=redis_conn)

@app.route('/')
def index():
    random = randint(1, 1000)
    return f'<h1>The number is: {random}</h1>'

@app.route('/scorecard/<question>', methods=['GET'])
def chatbot(question):
    try:
        job = q.enqueue(process_question, question)
        while not job.is_finished:
            time.sleep(1)
        # Wait for job to finish and get the result
        result = job.result

        # Check if result is None
        if result is not None:
            return jsonify({'output': result})
        else:
            return jsonify({"error": "Job result is None"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/process_question', methods=['POST'])
def run_process_question():
    data = request.get_json()
    if not data or 'question' not in data:
        return jsonify({'error': 'Invalid input'}), 400

    question = data['question']
    task = process_question.apply_async(args=[question])
    return jsonify({'task_id': task.id}), 202

@app.route('/result/<task_id>', methods=['GET'])
def get_result(task_id):
    task = AsyncResult(task_id)
    if task.state == 'PENDING':
        response = {
            'state': task.state,
            'status': 'Pending...'
        }
    elif task.state != 'FAILURE':
        response = {
            'state': task.state,
            'result': task.result
        }
    else:
        response = {
            'state': task.state,
            'status': str(task.info)
        }
    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv("PORT", 5050)), debug=True)
