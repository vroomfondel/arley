# FROM python:3.12-bookworm AS builder

ARG python_version=3.14
ARG debian_version=trixie

FROM python:${python_version}-${debian_version} AS builder

# need to repeat args (without defaults) in this stage
ARG python_version
ARG debian_version

RUN apt update && \
    apt -y full-upgrade && \
    apt -y install --no-install-recommends htop procps iputils-ping locales vim tini gcc && \
    pip install --upgrade pip && \
    rm -rf /var/lib/apt/lists/*

RUN sed -i -e 's/# de_DE.UTF-8 UTF-8/de_DE.UTF-8 UTF-8/' /etc/locale.gen && \
    locale-gen && \
    update-locale LC_ALL=de_DE.UTF-8 LANG=de_DE.UTF-8 && \
    rm -f /etc/localtime && \
    ln -s /usr/share/zoneinfo/Europe/Berlin /etc/localtime


# MULTIARCH-BUILD-INFO: https://itnext.io/building-multi-cpu-architecture-docker-images-for-arm-and-x86-1-the-basics-2fa97869a99b
ARG TARGETOS
ARG TARGETARCH
RUN echo "I'm building BUILDER-stage for $TARGETOS/$TARGETARCH"

# https://docs.docker.com/develop/develop-images/dockerfile_best-practices/

ARG UID=1234
ARG GID=1234
ARG UNAME=pythonuser
RUN groupadd -g ${GID} -o ${UNAME} && \
    useradd -m -u ${UID} -g ${GID} -o -s /bin/bash ${UNAME} && \
    mkdir /python_venv && chown ${UID}:${GID} /python_venv

USER ${UNAME}

WORKDIR /app
COPY --chown=${UID}:${GID} requirements.txt /

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN python3 -m venv /python_venv && . /python_venv/bin/activate && pip3 install --no-cache-dir --upgrade -r /requirements.txt

# final stage
FROM python:${python_version}-slim-${debian_version} AS finalimage
# see below for copy from builder-stage

# need to repeat args (without defaults) in this stage
ARG python_version
ARG debian_version

RUN apt update && \
    apt -y full-upgrade && \
    apt -y install --no-install-recommends htop procps iputils-ping locales vim tini gcc && \
    pip install --upgrade pip && \
    rm -rf /var/lib/apt/lists/*

RUN sed -i -e 's/# de_DE.UTF-8 UTF-8/de_DE.UTF-8 UTF-8/' /etc/locale.gen && \
    locale-gen && \
    update-locale LC_ALL=de_DE.UTF-8 LANG=de_DE.UTF-8 && \
    rm -f /etc/localtime && \
    ln -s /usr/share/zoneinfo/Europe/Berlin /etc/localtime


# MULTIARCH-BUILD-INFO: https://itnext.io/building-multi-cpu-architecture-docker-images-for-arm-and-x86-1-the-basics-2fa97869a99b
ARG TARGETOS
ARG TARGETARCH
RUN echo "I'm building FINAL-stage for $TARGETOS/$TARGETARCH"

ARG UID=1234
ARG GID=1234
ARG UNAME=pythonuser
RUN groupadd -g ${GID} -o ${UNAME} && \
    useradd -m -u ${UID} -g ${GID} -o -s /bin/bash ${UNAME}

USER ${UNAME}

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY --from=builder --chown=${UID}:${GID} /python_venv /python_venv

COPY --chown=${UID}:${GID} python_venv.sh /app/
COPY --chown=${UID}:${GID} arley /app/arley
COPY --chown=${UID}:${GID} dbaccess /app/dbaccess
COPY --chown=${UID}:${GID} main.py /app/

ARG gh_ref=gh_ref_is_undefined
ENV GITHUB_REF=$gh_ref
ARG gh_sha=gh_sha_is_undefined
ENV GITHUB_SHA=$gh_sha
ARG buildtime=buildtime_is_undefined
ENV BUILDTIME=$buildtime

# https://hynek.me/articles/docker-signals/
STOPSIGNAL SIGINT
#ENTRYPOINT ["/usr/bin/tini", "--"]

# ENV TINI_SUBREAPER=yes
# ENV TINI_KILL_PROCESS_GROUP=yes
# ENV TINI_VERBOSITY=3

ENTRYPOINT ["tini", "--"]

CMD ["tail", "-f", "/dev/null"]
