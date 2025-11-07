FROM python:3.12.12-slim

COPY server/* /app

WORKDIR /app
RUN pip install -r requirements.txt

EXPOSE 5000
ENTRYPOINT [ "python", "main.py" ]