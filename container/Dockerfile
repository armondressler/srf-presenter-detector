FROM python:3.10

RUN apt-get update -y && \
    apt-get upgrade -y && \
    apt-get install -y tesseract-ocr python3-pip ffmpeg && \
    rm -rf /var/lib/apt/lists/*

COPY ./files/app/requirements.txt /requirements.txt
RUN /usr/local/bin/python -m pip install --no-cache-dir --upgrade -r /requirements.txt

COPY ./files/app /app
COPY ./files/model /model

RUN groupadd -g 1000 uvicorn && \
    useradd -m -u 1000 -g uvicorn uvicorn && \
    chown -R uvicorn:uvicorn /app /model

USER uvicorn

WORKDIR /app

EXPOSE 8080
	
CMD ["python3", "main.py", "--model-filepath", "/model/tagesschau_unet.h5"]
