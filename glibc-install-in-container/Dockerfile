FROM ubuntu:20.04

WORKDIR /root

RUN apt update
RUN apt install -y build-essential python3 gawk bison git

COPY glibc-install.sh /root/glibc-install.sh
RUN /root/glibc-install.sh
ENV LD_LIBRARY_PATH=/root/glibc-install/lib:/lib/x86_64-linux-gnu
