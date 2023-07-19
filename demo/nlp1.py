import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
import yaml
import requests
import random
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import webbrowser
import requests
from bs4 import BeautifulSoup
import re
import datetime
import geocoder
import datetime
from mysql_data import store_reminder, get_reminders, get_mysql_connection 
from googlesearch import search
import requests
from bs4 import BeautifulSoup

def load_intents(intent_file):
    with open(intent_file, 'r') as file:
        intents = yaml.safe_load(file)
    return intents

def load_responses(response_file):
    with open(response_file, 'r') as file:
        responses = yaml.safe_load(file)
    return responses

intents = load_intents('data/intent.yml')
responses = load_responses('data/response.yml')

def preprocess_input(user_input):
    # Tokenize the input
    tokens = word_tokenize(user_input.lower())

    # Remove special characters and keep only alphabetic words
    tokens = [re.sub(r'[^a-zA-Z]', '', token) for token in tokens if re.sub(r'[^a-zA-Z]', '', token)]

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    # Lemmatize the tokens
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    return tokens

def get_intent(user_input):
    preprocessed_input = preprocess_input(user_input)
    best_score = 0
    best_intent = None

    for intent in intents:
        intent_examples = intent['examples']
        if isinstance(intent_examples, str):  # Handle a single example
            intent_examples = [intent_examples]

        for example in intent_examples:
            intent_tokens = preprocess_input(example)
            score = len(set(preprocessed_input).intersection(intent_tokens))

            if score > best_score:
                best_score = score
                best_intent = intent['intent']

    return best_intent

def get_response(intent, user_input):
    responses_list = [response['responses'] for response in responses if response['intent'] == intent]
    if responses_list:
        response = random.choice(responses_list)
        if '{query}' in response:
            return response.replace('{query}', user_input)
        else:
            return response
    else:
        return "I'm sorry, I didn't understand your request."

def get_current_location():
    g = geocoder.ip('me')
    current_location = g.city
    return current_location

def get_current_time():
    current_time = datetime.datetime.now().strftime("%H:%M:%S")
    return current_time

def get_current_date():
    current_date = datetime.date.today().strftime("%Y-%m-%d")
    return current_date


def scrape_google():
    user_query = input("AI Bot: What would you like to search for? ")
    search_results = list(search(user_query, num_results=5))

    if search_results:
        result_str = "AI Bot: Here are some top search results for " + user_query + ":\n"
        for result in search_results:
            result_str += "Link: " + result + "\n"
        return result_str
    else:
        return "AI Bot: Sorry, no search results were found for " + user_query




def scrape_wikipedia():
    query = input("AI Bot: What would you like to search for on Wikipedia? ")
    processed_query = str(query).replace(' ', '_').replace('[','').replace(']','')  # Replace spaces with underscores
    url = f"https://en.wikipedia.org/wiki/{processed_query}"

    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')

        paragraphs = soup.find_all('p')

        if paragraphs:
            num_lines = 2  # Number of lines to retrieve
            extracted_text = ""
            for paragraph in paragraphs[:num_lines]:
                extracted_text += paragraph.get_text(separator=' ')
            return extracted_text
        else:
            return "No text found on the page"
    except requests.exceptions.RequestException:
        return "Failed to retrieve data from Wikipedia."


# Function to check for reminders and update them
def check_reminders():
    # Establish a connection to the MySQL database
    connection = get_mysql_connection()

    if connection:
        try:
            # Create a cursor object to execute SQL queries
            cursor = connection.cursor()

            # Get the current time
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Fetch reminders that are active and have passed their datetime
            sql = "SELECT * FROM reminders WHERE status = 'Active' AND datetime <= %s"
            cursor.execute(sql, (current_time,))
            reminders = cursor.fetchall()

            if reminders:
                for reminder in reminders:
                    print("Reminder:", reminder[2])  # Print the reminder message
                    # Update the reminder status to inactive
                    update_sql = "UPDATE reminders SET status = 'Inactive' WHERE id = %s"
                    cursor.execute(update_sql, (reminder[0],))
                    connection.commit()

            # Close the cursor and database connection
            cursor.close()
            connection.close()

        except Error as e:
            print(f"Error checking reminders in MySQL: {e}")

    else:
        print('Failed to establish MySQL connection.')

# Function to check for reminders and display them
def check_reminder():
    reminders = get_reminders()
    current_time = datetime.datetime.now()
    #current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for reminder in reminders:
        if reminder['datetime'] >= current_time and reminder['status'] == "Active":
            print("Reminder:", reminder['message'])
            #reminder['status'] = "Closed"  # Mark the reminder as closed
            # Perform any additional action for the reminder here

    # Update the reminders in the database
    update_reminders(reminders)


# Function to update the reminders in the database
def update_reminders(reminders):
    # Establish a connection to the MySQL database
    connection = get_mysql_connection()

    if connection:
        try:
            # Create a cursor object to execute SQL queries
            cursor = connection.cursor()

            # Prepare the SQL statement to update the reminders
            sql = 'UPDATE reminders SET status = %s WHERE datetime = %s'

            # Update each reminder in the database
            for reminder in reminders:
                values = (reminder['status'], reminder['datetime'])
                cursor.execute(sql, values)

            # Commit the changes to the database
            connection.commit()

            # Close the cursor and database connection
            cursor.close()
            connection.close()

        except Error as e:
            print(f"Error updating reminders in MySQL: {e}")

    else:
        print('Failed to establish MySQL connection.')


def check_missed_messages():
    # Logic to check missed messages and handle reminders
    reminders = get_reminders()
    current_time = datetime.datetime.now()

    for reminder in reminders:
        reminder_datetime = reminder['datetime']

        if reminder_datetime <= current_time and reminder['status'] == "Active":
            print("Reminder:", reminder['message'])
            reminder['status'] = "Closed"  # Mark the reminder as closed
            # Perform any additional action for the reminder here

    # Update the reminders in the database
    update_reminders(reminders)


# Function to set a reminder
def set_reminder(user_input):
    reminder_datetime = input("Enter reminder date and time (YYYY-MM-DD HH:MM:SS): ")
    reminder_message = input("Enter reminder message: ")
    reminder_status = "Active"
    reminder = {'datetime': reminder_datetime, 'message': reminder_message, 'status': reminder_status}
    store_reminder(reminder)
    print("Reminder stored successfully.")
 
def check_missed_messages():
    # Logic to check missed messages and handle reminders
    reminders = get_reminders()
    for reminder in reminders:
        if reminder['status'] == "Active":
            # Call the app to check for missed messages for each reminder
            # Handle the missed messages and reminders based on the user's response
            print("Checking missed messages for reminder:", reminder['message'])
            # Perform any additional action for missed messages and reminders here
if __name__ == "__main__":
    while True:
        user_input = input("User: ")

        if user_input.lower() == "exit":
            break

        if user_input.lower() == "set a reminder":
            set_reminder(user_input)
        elif user_input.lower() == "check reminders":
            check_reminder()
        elif user_input.lower() == "check missed messages":
            check_missed_messages()
        elif user_input.lower() == "search" or user_input.lower() == "browse":
            scrape_google()
        elif user_input.lower() == "wiki" or user_input.lower() == "search wiki":
            scrape_wikipedia()    
        else:
            intent = get_intent(user_input)

            if intent in ["ask_time", "ask_day", "get_current_location","set_reminder","browse"]:
                if intent == "ask_time":
                    current_time = get_current_time()
                    print("AI Bot: The current time is", current_time)
                elif intent == "ask_day":
                    current_date = get_current_date()
                    print("AI Bot: Today's date is", current_date)
                elif intent == "get_current_location":
                    current_location = get_current_location()
                    print("AI Bot: Your current location is", current_location)
                elif intent == "set_reminder":
                    set_reminder(user_input)  
                elif intent == "browse":
                    scrape_google()         
            else:
                response = get_response(intent, user_input)
                print("AI Bot:", response)
