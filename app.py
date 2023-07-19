from flask import Flask, render_template, request , jsonify
from chatbot import  get_response
#from reminder import set_reminder
#from voice_assistent import get_response
app = Flask(__name__)

@app.get('/')
def home():
    return render_template('base.html')

# Dictionary to store session information
sessions = {}
@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.get_json().get('message')
    session_id = request.get_json().get('session_id')

    # Check if the user has an active session
    if session_id in sessions:
        # Retrieve session data
        session_data = sessions[session_id]
    else:
        # Initialize new session data
        session_data = {}

    # Process user input and generate response based on session data
    response = get_response(user_input, session_data)

    # Update session data
    session_data['previous_input'] = user_input
    session_data['previous_response'] = response

    # Store or update session data in the sessions dictionary
    sessions[session_id] = session_data

    # Check if the response requires user input
    if "set_reminder" in response:
        return jsonify(response)

    message = {"answer": response}
    return jsonify(message)
# @app.route("/chat", methods=["POST"])
# def chat():
#     user_input = request.get_json().get('message')
#     #session_id = request.get_json().get('session_id')
#     response = get_response(user_input)

#     # Check if the response requires user input
#     if "set_reminder" in response:
#         return jsonify(response)

#     message = {"answer": response}
#     return jsonify(message)


if __name__ == '__main__':
    app.run(debug=True)
