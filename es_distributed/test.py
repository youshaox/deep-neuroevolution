from email.mime.multipart import MIMEMultipart
import smtplib
from email.mime.text import MIMEText
from email.header import Header
import time
import datetime
import requests

SENDER_EMAIL = "Sometimesnaive@126.com"
SENDER_PASSWORD = 'ckRYMMHF6twhCMsm'

def send_email(sender_email, sender_password, starttime, endtime):
    reciever_email = 'shawnxys2016@gmail.com'
    cc_email = list().append(sender_email)
    message = MIMEMultipart()
    message['From'] = sender_email
    message['cc'] = cc_email
    message['To'] = reciever_email
    ts = time.time()
    time_suffix = str(datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H'))
    subject = 'Research实验结果: NSR-ES ' + time_suffix
    message['Subject'] = Header(subject, 'utf-8')

    # 邮件正文内容
    difference = endtime-starttime
    time_spending = str(divmod(difference.days * 86400 + difference.seconds, 60))
    try:
        hostname = requests.request('GET', 'http://myip.dnsomatic.com').text
    except Exception as e:
        from requests import get
        hostname = get('https://api.ipify.org').text
    email_context = """
    服务器{}实验已经完成。
    开始时间: {}
    结束时间: {}
    耗费时间: {}
    """.format(hostname, starttime, endtime, time_spending)
    message.attach(MIMEText(email_context, 'plain', 'utf-8'))
    server = smtplib.SMTP_SSL("smtp.126.com", 465)  # 发件人邮箱中的SMTP服务器，端口是25
    server.login(sender_email, sender_password)
    server.sendmail(sender_email, [reciever_email, ], message.as_string())  # 括号中对应的是发件人邮箱账号、收件人邮箱账号、发送邮件

starttime = datetime.datetime.now()
# time.sleep(2)
endtime = datetime.datetime.now()

send_email(SENDER_EMAIL, SENDER_PASSWORD, starttime, endtime)