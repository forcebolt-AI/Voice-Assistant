import random
import json
import datetime
import torch
import geocoder
from googlesearch import search
from nlu_nlp.model import NeuralNet
import nltk
from nlu_nlp.nltk_1 import bag_of_words, tokenize
from nlu_nlp.reminder import set_reminder , check_missed_messages , check_reminder
import requests
from bs4 import BeautifulSoup
import pandas as pd
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('data/intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Sam"

def get_response(msg, session_data):
    # Tokenize the message
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)  

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    function_mapping = {
        'get_current_location': get_current_location,
        'ask_time': get_current_time,
        'ask_day': get_current_date,
        'browse': lambda session_data: scrape_google(msg,session_data),
        'set_reminder': lambda session_data: set_reminder(msg,session_data),
        'check_reminder': check_reminder,
        'check_missed_reminder': check_missed_messages,
        'wiki': lambda session_data: scrape_wikipedia(msg,session_data),
        'joke': get_joke 
    }

    if tag in function_mapping:
        # Execute the function with session-specific data
        return function_mapping[tag](session_data)
    else:
        for intent in intents['intents']:
            if tag == intent['tag']:
                return random.choice(intent['responses'])

    return "I'm sorry, I didn't understand that."




def get_current_location(session_data):
    g = geocoder.ip('me')
    current_location = g.city
    return current_location


def get_current_time(session_data):
    current_time = datetime.datetime.now().strftime("%H:%M:%S")
    return current_time


def get_current_date(session_data):
    current_date = datetime.date.today().strftime("%Y-%m-%d")
    return current_date


def scrape_google(user_query, session_data):
    # Retrieve the desired number of search results from session data or use a default value
    num_results = session_data.get('num_results', 5)

    # Retrieve additional session-specific data
    additional_info = session_data.get('additional_info', None)

    # Perform the search using a library or API
    search_results = list(search(user_query, num_results=num_results))

    if search_results:
        result_str = "AI Bot: Here are some top search results for " + user_query + ":\n"
        for result in search_results:
            result_str += "Link: " + result + "\n"

        # Include additional session-specific data in the response
        if additional_info:
            result_str += "Additional Info: " + additional_info + "\n"

        return result_str
    else:
        return "AI Bot: Sorry, no search results were found for " + user_query



# def scrape_wikipedia(user_input,num_lines=3):
#     start_index = user_input.index("about") + len("about") + 1
#     query = user_input[start_index:]

#     processed_query = str(query).replace(' ', '_').replace('[', '').replace(']', '')
#     url = f"https://en.wikipedia.org/wiki/{processed_query}"

#     try:
#         response = requests.get(url)
#         soup = BeautifulSoup(response.content, 'html.parser')

#         paragraphs = soup.find_all('p')

#         if paragraphs:
#             #num_lines = 3
#             extracted_text = ""
#             for paragraph in paragraphs[:num_lines]:
#                 extracted_text += paragraph.get_text(separator=' ')
#             return extracted_text
#         else:
#             return "No text found on the page"
#     except requests.exceptions.RequestException:
#         return "Failed to retrieve data from Wikipedia."

def scrape_wikipedia(user_input, session_data):
    try:
        if "about" in user_input:
            start_index = user_input.index("about") + len("about") + 1
            query = user_input[start_index:]
        else:
            query = user_input

        processed_query = str(query).replace(' ', '_').replace('[', '').replace(']', '')
        url = f"https://en.wikipedia.org/wiki/{processed_query}"

        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')

        paragraphs = soup.find_all('p')

        if paragraphs:
            extracted_text = ""
            line_count = 0
            for paragraph in paragraphs:
                paragraph_text = paragraph.get_text(separator=' ')
                if paragraph_text.strip():
                    extracted_text += paragraph_text + "\n"
                    line_count += 1
                    if line_count >= 3:
                        break
            return extracted_text.strip()
        else:
            return "No text found on the page"
    except requests.exceptions.RequestException:
        return "Failed to retrieve data from Wikipedia."
    except ValueError:
        return "Invalid input for Wikipedia search."



def get_joke(session_data):
    # Read the CSV file using pandas
    df = pd.read_csv('shortjokes.csv')

    # Convert the DataFrame to a numpy array
    jokes = df.to_numpy()

    # Select a random joke
    random_joke = random.choice(jokes)

    # Generate a response with the random joke
    response = "Sure! Here's a joke for you:\n\n"
    response += f"Joke # {random_joke[1]}"

    return response



if __name__ == "__main__":
    print("Let's chat! (type 'quit' to exit)")
    session_id = 1  # Unique identifier for each session
    session_data = {}  # Initialize an empty session data dictionary
    session_id = 1  # Unique identifier for each session

    while True:
        sentence = input("You: ")
        if sentence == "quit":
            break

        resp = get_response(sentence , session_data)
        print(resp)
        if "Additional Info:" in resp:
            session_data["additional_info"] = "more info"
