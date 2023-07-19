import re
import datetime
from dateutil.parser import parse
from nlu_nlp.mysql_data import get_mysql_connection , store_reminder , check_reminders ,get_reminders
import requests
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
            sql = "SELECT * FROM store_reminder WHERE status = 'Active' AND datetime <= %s"
            cursor.execute(sql, (current_time,))
            reminders = cursor.fetchall()

            if reminders:
                for reminder in reminders:
                    print("Reminder:", reminder[2])  # Print the reminder message
                    # Update the reminder status to inactive
                    update_sql = "UPDATE store_reminder SET status = 'Inactive' WHERE id = %s"
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
    current_time = datetime.date.today()
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
            sql = 'UPDATE store_reminder SET status = %s WHERE datetime = %s'

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


# # Function to set a reminder
# def set_reminder(user_input):
#     reminder_datetime = input("Enter reminder date and time (YYYY-MM-DD HH:MM:SS): ")
#     reminder_message = input("Enter reminder message: ")
#     reminder_status = "Active"
#     reminder = {'datetime': reminder_datetime, 'message': reminder_message, 'status': reminder_status}
#     store_reminder(reminder)
#     print("Reminder stored successfully.")
 

def set_reminder(pattern,session_data):
    try:
        # Extract reminder time and message from the pattern using regular expressions
        #time_match = re.search(r"\b\d{1,2}:\d{2}\b", pattern)
        message_match = re.search(r"to (.+)", pattern)
        #r"remind me for (.+) on (\d{1,2}) at (\d{1,2}:\d{2})
        if message_match:
            #reminder_time = time_match.group()
            reminder_message = message_match.group(1)

            # Extract reminder date from the pattern
            date_text = re.search(r"\bon (.+?)\b", pattern)
            time_match = re.search(r"\b\d{1,2}:\d{2}\b", pattern)
            if date_text and time_match:
                reminder_date = parse(date_text.group(1), fuzzy=True).date()
                reminder_time = parse(time_match.group(), fuzzy=True).time()
            else:
                reminder_date = None
                reminder_time = None
                #reminder_date = datetime.today()
            #reminder_date = reminder_date.datetime.date()
            #reminder_time = reminder_time.datetime.time()
            reminder_status = "Active"
            reminder = {'datetime': reminder_date, "reminder_time": reminder_time, 'message': reminder_message, 'status': reminder_status}
            store_reminder(reminder)
            return f"Sure i will remind you for {reminder_message}"
            # Prompt the user to confirm storing the reminder
            response = input("Are you sure you want to store this reminder? ")
            if response.lower() == "yes":
                store_reminder(reminder)
                return "Reminder stored successfully."
            else:
                return "Reminder not stored."
        else:
            return "Unable to extract reminder details from the pattern."

    except KeyboardInterrupt:
        return "Reminder setting interrupted. No reminder was stored."
