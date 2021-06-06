FROM debian:buster
LABEL maintainer="Abhishek N. Kulkarni" \
        email="abhi.bp1993@gmail.com" \
        version="0.0.1"


# Install python
RUN apt-get update && \
    RUNLEVEL=1 DEBIAN_FRONTEND=noninteractive \
    apt-get install -y --allow-unauthenticated --no-install-recommends \
    build-essential cmake \
    python3-pip python3-setuptools python3-dev pkg-config  && \
    apt-get clean

# Install spot
# Reference: https://gitlab.lrde.epita.fr/spot/spot-web/-/blob/master/docker/Dockerfile
RUN echo 'deb [trusted=true] http://www.lrde.epita.fr/repo/debian/ stable/' >> /etc/apt/sources.list && \
    apt-get update && \
    RUNLEVEL=1 DEBIAN_FRONTEND=noninteractive \
    apt-get install -y --allow-unauthenticated --no-install-recommends \
    spot libspot-dev spot-doc python3-spot && \
    apt-get clean

RUN apt-get update && \
    RUNLEVEL=1 DEBIAN_FRONTEND=noninteractive \
    apt-get install -y --allow-unauthenticated --no-install-recommends \
    graphviz graphviz-dev && \
    apt-get clean

# Install python packages
RUN pip3 install numpy matplotlib pyzmq pydot networkx pygraphviz pytest 

# Patch to avoid ERROR: (Could not install packages due to an EnvironmentError: [SSL: DECRYPTION_FAILED_OR_BAD_RECORD_MAC] decryption failed or bad record mac (_ssl.c:2488))
RUN apt-get install -y --no-install-recommends ca-certificates
RUN update-ca-certificates
RUN pip3 install scipy pygame 
RUN pip3 install pyyaml 
RUN pip3 install lark