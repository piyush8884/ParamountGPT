import smtplib
import os
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from email.mime.text import MIMEText
from dotenv import load_dotenv
import schedule
import time

# Load environment variables
load_dotenv()

# Email configuration
EMAIL_ADDRESS = os.getenv("YOUR_EMAIL_ADDRESS")  # Your email address
EMAIL_PASSWORD = os.getenv("YOUR_EMAIL_PASSWORD")  # App password generated from Gmail
TO_EMAIL = os.getenv("TO_EMAIL")  # Recipient email address

# Path to the log file
LOG_FILE_PATH = "app/Logs/app.log"

def send_email():
    try:
        # Create the email message
        msg = MIMEMultipart()
        msg['From'] = EMAIL_ADDRESS
        msg['To'] = TO_EMAIL
        msg['Subject'] = "App Log File - Hourly Update"

        # Add body text
        body = "Please find the attached app log file."
        msg.attach(MIMEText(body, 'plain'))

        # Attach the log file
        with open(LOG_FILE_PATH, "rb") as log_file:
            part = MIMEBase('application', 'octet-stream')
            part.set_payload(log_file.read())
            encoders.encode_base64(part)
            part.add_header('Content-Disposition', f"attachment; filename= {os.path.basename(LOG_FILE_PATH)}")
            msg.attach(part)

        # Send the email
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.sendmail(EMAIL_ADDRESS, TO_EMAIL, msg.as_string())
            print("Email sent successfully.")

    except Exception as e:
        print(f"Failed to send email: {e}")

# Schedule the email to be sent every hour
schedule.every(2).hour.do(send_email)
# schedule.every(2).minutes.do(send_email)
print("Scheduler started. Sending app.log every hour.")

# Run the scheduler
while True:
    schedule.run_pending()
    time.sleep(60)  # Check every minute
