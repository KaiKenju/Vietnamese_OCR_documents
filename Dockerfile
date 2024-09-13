FROM ubuntu
FROM python:3.10

WORKDIR /src


COPY multi_page.py ./multi_page.py
COPY requirements.txt ./requirements.txt


RUN apt-get update && apt-get -y upgrade
RUN apt-get -y install python3 
RUN pip install -r requirements.txt

CMD ["python3", "multi_page.py"]