# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 20:24:59 2024

@author: kapta
"""

import smtplib

smtp_server = 'smtp.gmail.com'
smtp_port = 587
smtp_username = 'kaptain2886@gmail.com'
smtp_password = 'dgzk ndch ippd qpyl'

from_email = 'kaptain2886@gmail.com'
to_email = '305105457@qq.com'
subject = 'Hello, world!'
body = 'This is a test email.'

message = f'Subject: {subject}\n\n{body}'

with smtplib.SMTP(smtp_server, smtp_port) as smtp:
    smtp.starttls()
    smtp.login(smtp_username, smtp_password)
    smtp.sendmail(from_email, to_email, message)