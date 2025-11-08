FROM instrumentisto/flutter:3 AS builder

WORKDIR /flutter_stock_analyze
COPY flutter_stock_analyze .
RUN flutter build web

FROM python:3.12-alpine

ENV TZ=Asia/Shanghai
RUN sed -i 's#https\?://dl-cdn.alpinelinux.org/alpine#https://mirrors.tuna.tsinghua.edu.cn/alpine#g' /etc/apk/repositories
RUN apk update && apk add --no-cache gcc make libc-dev linux-headers pcre-dev \
    && mkdir /application
# RUN mkdir /app
COPY server application
COPY --from=builder /flutter_stock_analyze/build/web /application/app
WORKDIR /application
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple \
    && pip install -r requirements.txt && pip install uwsgi
# RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && pip install -r requirements.txt && pip install uwsgi

EXPOSE 5000
ENTRYPOINT [ "uwsgi", "--ini", "uwsgi.ini" ]