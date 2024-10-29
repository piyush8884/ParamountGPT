import os
import csv

def store_feedback(feedback_data, filename='feedback.csv'):
    fieldnames = ['query', 'response', 'rating']
    if not os.path.isfile(filename):
        with open(filename, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()

    with open(filename, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writerow(feedback_data)
