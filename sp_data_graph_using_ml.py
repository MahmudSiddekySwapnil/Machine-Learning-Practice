import mysql.connector
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Connect to MySQL database
db_connection = mysql.connector.connect(
    host="127.0.0.1",
    user="root",
    password="Password123#@!",
    database="sp_admin_db"
)

# Check if the connection is successful
if db_connection.is_connected():
    print("Connected to MySQL database")

    # Define the start and end dates for the period of interest
    start_date = datetime(2023, 1, 1)  # January 1st, 2024
    end_date = datetime(2024, 7, 30)   # July 30th, 2024

    # Query example: Fetching day-wise transaction amounts between specified dates
    cursor = db_connection.cursor()
    query = """
        SELECT DATE(created_at) AS transaction_date, SUM(amount_recived) AS total_amount
        FROM trx_history
        WHERE created_at >= %s AND created_at <= %s
        AND sp_code = 1000
        GROUP BY transaction_date
        ORDER BY transaction_date
    """
    cursor.execute(query, (start_date, end_date))

    # Fetch data for day-wise transaction amounts
    data = cursor.fetchall()

    # Prepare data for day-wise plotting
    dates = []
    amounts = []
    for row in data:
        dates.append(row[0])   # Date
        amounts.append(row[1])  # Total amount for the date

    # Plot day-wise total transaction amounts
    plt.figure(figsize=(16, 8))  # Adjust figure size for better visibility
    plt.subplot(2, 1, 1)  # 2 rows, 1 column, plot 1
    plt.bar(dates, amounts, align='center', alpha=0.5)
    plt.xlabel('Date')
    plt.ylabel('Total Transaction Amount')
    plt.title('Day-wise Total Transaction Amounts (Jan-Jul 2024)')
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Fetch data for monthly transaction amounts
    cursor.execute("""
        SELECT DATE_FORMAT(created_at, '%Y-%m') AS month_year, SUM(amount_recived) AS total_amount
        FROM trx_history
        WHERE created_at >= %s AND created_at <= %s
        GROUP BY month_year
        ORDER BY month_year
    """, (start_date, end_date))

    data_monthly = cursor.fetchall()

    # Prepare data for monthly plotting
    months = []
    amounts_monthly = []
    for row in data_monthly:
        months.append(row[0])   # Month-Year
        amounts_monthly.append(row[1])  # Total amount for the month

    # Determine the month with the highest total transaction amount
    max_amount = max(amounts_monthly)
    max_month_index = amounts_monthly.index(max_amount)
    max_month = months[max_month_index]

    # Plot monthly total transaction amounts highlighting the highest month
    plt.subplot(2, 1, 2)  # 2 rows, 1 column, plot 2
    bars = plt.bar(months, amounts_monthly, align='center', alpha=0.5, color='red')

    # Highlight the bar for the highest month
    bars[max_month_index].set_color('red')

    plt.xlabel('Month')
    plt.ylabel('Total Transaction Amount')
    plt.title('Monthly Total Transaction Amounts (Jan-Jul 2024)')
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Display the month with the highest total transaction amount
    print(f"The month with the highest total transaction amount: {max_month} (Amount: {max_amount})")

    # Show plots
    plt.show()

    # Close cursor and database connection
    cursor.close()
    db_connection.close()
    print("MySQL connection is closed")
else:
    print("Failed to connect to MySQL database")
