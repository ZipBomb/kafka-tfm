FROM python:3
COPY . /app
WORKDIR /app
RUN pip install kafka-python
RUN pip install graphitesender
CMD python ./to_graphite.py