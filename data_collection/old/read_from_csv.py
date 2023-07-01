import csv
import json

# Open the CSV file
with open('publications.csv', 'r') as file:
    # Create a CSV reader
    csv_reader = csv.reader(file)

    # Iterate over each row in the CSV file
    for row in csv_reader:
        print(row)
        # Extract the JSON object from the row
        json_object = json.loads(row[0])

        # Print the JSON object
        print(json_object)
