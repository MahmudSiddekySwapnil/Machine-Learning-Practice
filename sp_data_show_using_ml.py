from flask import Flask, jsonify
import mysql.connector
from datetime import datetime

app = Flask(__name__)

# MySQL database connection configuration
db_config = {
    'host': '127.0.0.1',
    'user': 'root',
    'password': 'Password123#@!',
    'database': 'sp_admin_db'
}

# Endpoint to fetch day-wise transaction amounts
@app.route('/api/day-wise-transactions', methods=['GET'])
def get_day_wise_transactions():
    try:
        # Connect to MySQL database
        db_connection = mysql.connector.connect(**db_config)

        # Define the start and end dates for the period of interest
        start_date = datetime(2023, 1, 1)  # January 1st, 2023
        end_date = datetime(2024, 7, 30)   # July 30th, 2024

        # Query to fetch day-wise transaction amounts
        query = """
            SELECT DATE(created_at) AS transaction_date, SUM(amount_recived) AS total_amount
            FROM trx_history
            WHERE created_at >= %s AND created_at <= %s
            GROUP BY transaction_date
            ORDER BY transaction_date
        """
        cursor = db_connection.cursor()
        cursor.execute(query, (start_date, end_date))
        data = cursor.fetchall()

        # Prepare response data
        day_wise_transactions = []
        for row in data:
            day_wise_transactions.append({
                'date': row[0].isoformat(),
                'total_amount': float(row[1])  # Convert to float if needed
            })

        cursor.close()
        db_connection.close()

        return jsonify(day_wise_transactions)

    except mysql.connector.Error as err:
        print(f"Error retrieving day-wise transactions: {err}")
        return jsonify({'error': 'Failed to fetch data'}), 500

# Endpoint to fetch monthly transaction amounts
@app.route('/api/monthly-transactions', methods=['GET'])
def get_monthly_transactions():
    try:
        # Connect to MySQL database
        db_connection = mysql.connector.connect(**db_config)

        # Define the start and end dates for the period of interest
        start_date = datetime(2023, 1, 1)  # January 1st, 2023
        end_date = datetime(2024, 7, 30)   # July 30th, 2024

        # Query to fetch monthly transaction amounts
        query = """
            SELECT DATE_FORMAT(created_at, '%Y-%m') AS month_year, SUM(amount_recived) AS total_amount
            FROM trx_history
            WHERE created_at >= %s AND created_at <= %s
            GROUP BY month_year
            ORDER BY month_year
        """
        cursor = db_connection.cursor()
        cursor.execute(query, (start_date, end_date))
        data = cursor.fetchall()

        # Prepare response data
        monthly_transactions = []
        for row in data:
            monthly_transactions.append({
                'month_year': row[0],
                'total_amount': float(row[1])  # Convert to float if needed
            })

        cursor.close()
        db_connection.close()

        return jsonify(monthly_transactions)

    except mysql.connector.Error as err:
        print(f"Error retrieving monthly transactions: {err}")
        return jsonify({'error': 'Failed to fetch data'}), 500

if __name__ == '__main__':
    app.run(debug=True)
