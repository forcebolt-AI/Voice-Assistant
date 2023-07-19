import mysql.connector
from mysql.connector import Error
from datetime import datetime

# Function to establish MySQL connection
def get_mysql_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="User@123",
        database="user"
    )

def store_reminder(reminder):
    # Establish a connection to the MySQL database
    connection = get_mysql_connection()

    if connection:
        try:
            # Create a cursor object to execute SQL queries
            cursor = connection.cursor()
            # Create the table if it doesn't exist
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS store_reminder (
            id INT AUTO_INCREMENT PRIMARY KEY,
            datetime DATETIME,
            reminder_time TIME,
            message VARCHAR(255),
            status VARCHAR(20)
        )
    """)
            # Prepare the SQL statement to insert a reminder
            sql = 'INSERT INTO store_reminder (datetime,reminder_time, message, status) VALUES (%s, %s, %s ,%s)'
            values = (reminder['datetime'],reminder['reminder_time'], reminder['message'], reminder['status'])

            # Execute the SQL statement
            cursor.execute(sql, values)

            # Commit the changes to the database
            connection.commit()

            # Close the cursor and database connection
            cursor.close()
            connection.close()

        except Error as e:
            print(f"Error storing reminder in MySQL: {e}")

    else:
        print('Failed to establish MySQL connection.')

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
# Function to retrieve reminders from the MySQL database
def get_reminders():
    reminders = []

    # Establish a connection to the MySQL database
    connection = get_mysql_connection()

    if connection:
        try:
            # Create a cursor object to execute SQL queries
            cursor = connection.cursor()

            # Execute the SQL statement to retrieve reminders
            cursor.execute('SELECT datetime, message, status FROM store_reminders')

            # Fetch all the rows and append them to the reminders list
            rows = cursor.fetchall()
            for row in rows:
                reminders.append({'datetime': row[0], 'message': row[1], 'status': row[2]})

            # Close the cursor and database connection
            cursor.close()
            connection.close()

        except Error as e:
            print(f"Error retrieving reminders from MySQL: {e}")

    else:
        print('Failed to establish MySQL connection.')

    return reminders

# def user_id():
#     connect = get_mysql_connection()
#     if connect:
#         try:
#             # Create a cursor object to execute SQL queries
#             cursor = connect.cursor()
#             # Create the table if it doesn't exist
#             cursor.execute("""
#             CREATE TABLE IF NOT EXISTS store_user (
#             id INT AUTO_INCREMENT PRIMARY KEY,
#             user_id ,
#             name ,
#             email
#         )
#     """)       
            




def get_user_data(user_id):
    # Connect to MySQL database
    cnx = get_mysql_connection()
    
    # Create a cursor object
    cursor = cnx.cursor()
    
    # Check if user exists in the database
    query = "SELECT * FROM users WHERE user_id = %s"
    cursor.execute(query, (user_id,))
    result = cursor.fetchone()
    
    if result:
        # User exists, retrieve their data
        user_data = {
            "user_id": result[0],
            "name": result[1],
            "email": result[2],
            # Add more columns as needed
        }
    else:
        # User does not exist, return None or raise an exception
        user_data = None
    
    # Close the cursor and database connection
    cursor.close()
    cnx.close()
    
    return user_data
