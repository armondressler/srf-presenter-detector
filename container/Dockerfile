FROM python:3.10

RUN apt-get update -y && \
    apt-get upgrade -y

COPY ./files/app/requirements.txt /requirements.txt
RUN /usr/local/bin/python -m pip install --no-cache-dir --upgrade -r /requirements.txt

COPY ./files/app /app

RUN groupadd -g 1000 uvicorn && \
    useradd -m -u 1000 -g uvicorn uvicorn

WORKDIR /app

EXPOSE 8080
	
CMD ["python3", "main.py"]
