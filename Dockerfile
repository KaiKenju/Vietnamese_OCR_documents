FROM python:3.10

WORKDIR /src


COPY app.py ./app.py
COPY requirements.txt ./requirements.txt


RUN apt-get update && apt-get -y upgrade
RUN apt-get -y install python3 
RUN pip install -r requirements.txt

EXPOSE 7860


CMD ["python3", "-u", "app.py", "--host", "0.0.0.0", "--port", "7860"]