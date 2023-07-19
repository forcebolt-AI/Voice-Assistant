from twilio.rest import Client
#store your twilio api in config file
from config import account_sid , auth_token
client = Client(account_sid, auth_token)

message = client.messages.create(
    body='Hello, this is a test message!',
    from_='twilio num',
    to='phone number'
)
# print(message.sid)
# call = client.calls.create(
#     twiml='<Response><Say>Hello, this is a test call!</Say></Response>',
#     from_='Your_Twilio_Phone_Number',
#     to='Recipient_Phone_Number'
# )
# print(call.sid)


from datetime import datetime
from twilio.rest import Client
import mysql.connector
from mysql_data import get_mysql_connection
def send_notifications():
    # Twilio credentials
    account_sid = 'ACd6a474562cd1e7038af4f2b8b30b4f69'
    auth_token = '2512418f04121fd8ec87f00fb93d893a'
    twilio_phone_number = '+14847597970'
    recipient_phone_number = '+917780633944'

    # Connect to MySQL database
    cnx = get_mysql_connection()

    # Create a cursor object
    cursor = cnx.cursor()

    # Fetch reminders from the database
    query = ("SELECT message, datetime FROM store_reminder")
    cursor.execute(query)

    # Retrieve reminders from the database
    rows = cursor.fetchall()
    for row in rows:
        message = row[0]
        reminder_datetime = row[1]

        # Compare the reminder datetime with the current datetime
        current_datetime = datetime.now()
        if reminder_datetime == current_datetime:
            # Send notification using Twilio
            client = Client(account_sid, auth_token)

            message = client.messages.create(
                body=f"Reminder: {message}",
                from_=twilio_phone_number,
                to=recipient_phone_number
            )

            # Print the sent message
            print(f"Sent reminder: {message}")

    # Close the cursor and database connection
    cursor.close()
    cnx.close()


# Example usage
send_notifications()
