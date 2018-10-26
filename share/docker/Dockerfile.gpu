FROM nvidia/cuda:9.0-devel

LABEL mantainer="Sebastian Funk <sebastian.funk@lshtm.ac.uk>"

USER root

# libbi prerequisittes
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    build-essential \
    libblas-dev \
    liblapack-dev \
    libqrupdate-dev \
    libboost-all-dev \
    libgsl0-dev \
    libnetcdf-dev \
    autoconf \
    automake \
    libthrust-dev \
 && BUILDDEPS="cpanminus \
    libmodule-install-perl \
    curl" \
 && apt-get install -y --no-install-recommends $BUILDDEPS \
 ## install libbi
 && curl -fLo /tmp/LibBi.tar.gz https://github.com/lawmurray/LibBi/archive/stable.tar.gz \
 && tar xzf /tmp/LibBi.tar.gz -C ~ \
 && cd ~/LibBi-stable \
 && cpanm --quiet --installdeps --notest . \
 && cpanm --quiet . \
 && cd .. \
 && rm /tmp/LibBi.tar.gz \
 && rm -rf ~/LibBi-stable \
 && apt-get remove --purge -y $BUILDDEPS \
 && apt-get autoremove -y \
 && apt-get autoclean -y \
 && rm -rf /var/lib/apt/lists/*
