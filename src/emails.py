# Helpers for sending notification emails using SendGrid.
# Requires 'sendgrid' package:
#   pip install sendgrid

import sys
import os

import sendgrid
from sendgrid.helpers import mail

def _read_key_file(fname):
    with open(fname) as fd:
        return fd.read().strip()

# Get API key (shared for workshop account)
SENDGRID_KEY_PATH="/nfs/jsalt/share/sendgrid.key"
SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY", None)
if SENDGRID_API_KEY is None:
    SENDGRID_API_KEY = _read_key_file(SENDGRID_KEY_PATH)

DEFAULT_SENDER=mail.Email("jsalt.sentence.rep.2018+notifier@gmail.com",
                          name="JSALT Sentence Representative")

def make_message(to: str, subject: str, body: str) -> mail.Mail:
    to_email = mail.Email(to)
    content = mail.Content("text/plain", body)
    return mail.Mail(DEFAULT_SENDER, subject, to_email, content)

def send_message(message: mail.Mail):
    sg = sendgrid.SendGridAPIClient(apikey=SENDGRID_API_KEY)
    response = sg.client.mail.send.post(request_body=message.get())
    return response
