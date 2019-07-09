# Helpers for sending notification emails using SendGrid.
# Requires 'sendgrid' package:
#   pip install sendgrid

import datetime
import logging as log
import os
import socket
import sys

import pytz
import sendgrid
from sendgrid.helpers import mail


def _read_key_file(fname):
    with open(fname) as fd:
        return fd.read().strip()


# Get API key (shared for workshop account)
SENDGRID_KEY_PATH = "/nfs/jsalt/share/sendgrid.key"
SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY", None)
if SENDGRID_API_KEY is None:
    SENDGRID_API_KEY = _read_key_file(SENDGRID_KEY_PATH)

DEFAULT_SENDER = mail.Email("jsalt.sentence.rep.2018+notifier@gmail.com", name="Cookie Monster")

LOCALTZ = pytz.timezone("US/Eastern")


def make_message(to: str, subject: str, body: str) -> mail.Mail:
    to_email = mail.Email(to)
    content = mail.Content("text/plain", body)
    return mail.Mail(DEFAULT_SENDER, subject, to_email, content)


def send_message(message: mail.Mail):
    sg = sendgrid.SendGridAPIClient(apikey=SENDGRID_API_KEY)
    response = sg.client.mail.send.post(request_body=message.get())
    return response


##
# Implementation-specific logic.


def get_notifier(to: str, args):
    """ Get a notification handler to call on exit.

    Args:
        to: recipient email address
        args: config.Params object, main config

    Returns:
        function(str, str), call with message body and subject prefix to send a
        notification email using sendgrid.
    """
    hostname = socket.gethostname()

    def _handler(body: str, prefix: str = ""):
        """ Email notifier. Sends an email. """
        # Construct subject line from args:
        subj_tmpl = "{prefix:s} '{exp_name:s}/{run_name:s}' on host '{host:s}'"
        prefix = prefix + " run" if prefix else "Run"
        subject = subj_tmpl.format(
            prefix=prefix, host=hostname, exp_name=args.exp_name, run_name=args.run_name
        )
        # Add timestamp.
        now = datetime.datetime.now(LOCALTZ)
        now = now.strftime("%Y-%m-%d %H:%M:%S")
        body = f"{now:s} {LOCALTZ.zone:s}\n\n" + body

        # Add log info.
        body += "\n\n Experiment log: {:s}".format(args.local_log_path)
        try:
            from . import gcp

            body += "\n Remote log (if enabled): " + gcp.get_remote_log_url(args.remote_log_name)
        except Exception as e:
            log.info("Unable to generate remote log URL - not on GCP?")

        # Add experiment args.
        body += "\n\n Parsed experiment args: {:s}".format(str(args))
        message = make_message(to, subject, body)
        log.info("Sending notification email to %s with subject: \n\t%s", to, subject)
        return send_message(message)

    return _handler
