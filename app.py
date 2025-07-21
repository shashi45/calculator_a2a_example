import os
import json
from flask import Flask, render_template, request, jsonify
from agent import CalculatorAgent

app = Flask(__name__)
agent = CalculatorAgent()

@app.route('/')
def chat():
    return render_template('chat.html')

@app.route('/api/chat', methods=['POST'])
def handle_chat():
    try:
        user_message = request.json.get('message', '')
        if not user_message:
            return jsonify({'error': 'No message provided'}), 400
        
        # Send message to agent and get response
        response = agent.process_message(user_message)
        
        return jsonify({
            'response': response['message'],
            'calculation': response.get('calculation'),
            'result': response.get('result')
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)