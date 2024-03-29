FROM python:3.7

RUN apt-get update && \
    apt-get install nginx -y && \
    pip3 install supervisor

COPY requirements.txt .
RUN pip3 install -r requirements.txt

RUN mkdir /app
WORKDIR /app

RUN mkdir /prom
ENV PROMETHEUS_MULTIPROC_DIR=/prom

COPY supervisord.conf /etc/supervisord.conf
COPY configuration.nginx /etc/configuration.nginx
COPY server.py /app/

ENTRYPOINT ["supervisord", "-c", "/etc/supervisord.conf"]
