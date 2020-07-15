# nginx-gunicorn-flaskRestful

FROM ubuntu:18.04

RUN apt-get update
RUN apt-get upgrade -y
RUN apt-get install -y python3 python3-pip python3-virtualenv nginx supervisor
RUN python3 -m pip install --no-cache gunicorn

ENV DEBIAN_FRONTEND="noninteractive"
ENV TZ="America/Los_Angeles"

# Setup flask application
RUN mkdir -p /deploy/app
COPY dashboard /deploy/app
RUN python3 -m pip install --no-cache -r /deploy/app/requirements.txt
RUN apt-get --assume-yes install python3-opencv

# Setup nginx
RUN rm /etc/nginx/sites-enabled/default
COPY flask.conf /etc/nginx/sites-available/
RUN ln -s /etc/nginx/sites-available/flask.conf /etc/nginx/sites-enabled/flask.conf
RUN echo "daemon off;" >> /etc/nginx/nginx.conf

# Setup supervisord
RUN mkdir -p /var/log/supervisor
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf
COPY gunicorn.conf /etc/supervisor/conf.d/gunicorn.conf

# For testing only
RUN mkdir -p /data/ian
RUN mkdir -p /data/ria
COPY test/experiments /data/ian/experiments
COPY test/experiments /data/ria/experiments
COPY test/grid-search /data/ian/grid-search
COPY test/grid-search /data/ria/grid-search

# Start processes
CMD ["/usr/bin/supervisord"]