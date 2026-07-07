FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV SGE_ROOT=/var/lib/gridengine
ENV SGE_CELL=default
ENV PATH="/usr/local/bin:/usr/bin:/bin:${PATH}"

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        ca-certificates \
        dnsutils \
        openssh-client \
    && rm -rf /var/lib/apt/lists/*

COPY host-root/ /
COPY entrypoint.sh /usr/local/bin/qsub-entrypoint

RUN sed -i 's/\r$//' /usr/local/bin/qsub-entrypoint \
    && chmod +x /usr/local/bin/qsub-entrypoint \
    && test -x /usr/local/bin/qsub

WORKDIR /work

ENTRYPOINT ["/usr/local/bin/qsub-entrypoint"]
CMD ["qsub --help"]
