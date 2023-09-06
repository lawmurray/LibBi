FROM ubuntu:22.04

LABEL maintainer="Sebastian Funk <sebastian.funk@lshtm.ac.uk>"

USER root

# libbi prerequisites
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    libblas-dev \
    liblapack-dev \
    libqrupdate-dev \
    libboost-all-dev \
    libgsl0-dev \
    libnetcdf-dev \
    ca-certificates \
    autoconf \
    automake \
    libthrust-dev \
    cpanminus \
    libmodule-install-perl \
    curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# install libbi
RUN curl -fLo /tmp/LibBi.tar.gz https://github.com/lawmurray/LibBi/archive/stable.tar.gz && \
    tar xzf /tmp/LibBi.tar.gz -C ~ && \
    cd ~/LibBi-stable && \
    cpanm --quiet --installdeps --notest . && \
    cpanm --quiet --force . && \
    cd .. && \
    rm /tmp/LibBi.tar.gz && \
    rm -rf ~/LibBi-stable

# Ensure apt runs in non-interactive mode
ARG DEBIAN_FRONTEND=noninteractive

# utils for GitHub Actions CI
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    npm \
    sudo \
    tzdata \
    locales

## Configure default locale,
## see https://github.com/rocker-org/rocker/issues/19
RUN echo "en_US.UTF-8 UTF-8" >> /etc/locale.gen && locale-gen en_US.utf8 && /usr/sbin/update-locale LANG=en_US.UTF-8
