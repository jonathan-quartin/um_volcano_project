conn = sqlite3.connect('rain_table.db')

cursor = conn.cursor()

create_table_query = '''
    CREATE TABLE IF NOT EXISTS rain_table (
        Date STRING,
        Longitude FLOAT,
        Latitude FLOAT,
        Precipitation FLOAT
    )
'''
cursor.execute(create_table_query)

# Replace 'your_csv_file.csv' with the path to your CSV file
with open('precipitation_galapagos.csv', 'r', newline='') as csv_file:
    csv_reader = csv.reader(csv_file)
    next(csv_reader)  # Skip the header row if it exists in the CSV

    for row in csv_reader:
        # Replace the table name and column names with your table and column names
        insert_query = '''
            INSERT INTO rain_table (Date, Longitude, Latitude, Precipitation)
            VALUES (?, ?, ?, ?)
        '''
        cursor.execute(insert_query, row)

# Commit the changes and close the connection
conn.commit()
conn.close()