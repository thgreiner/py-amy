FROM debian:latest

RUN apt-get update
RUN apt-get install -y curl gnupg busybox
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -
RUN echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | tee /etc/apt/sources.list.d/coral-edgetpu.list
RUN apt-get update
RUN apt-get install -y edgetpu-compiler
COPY quantized-model .

EXPOSE 8000
CMD busybox httpd -f -p 8000
