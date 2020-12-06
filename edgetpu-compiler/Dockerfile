FROM debian:latest

RUN apt-get update
RUN apt-get install -y curl gnupg
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -
RUN echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | tee /etc/apt/sources.list.d/coral-edgetpu.list
RUN apt-get update
RUN apt-get install -y edgetpu-compiler
RUN mkdir data

WORKDIR /data
