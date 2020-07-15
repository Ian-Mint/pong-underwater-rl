FROM conda/miniconda3

RUN mkdir -p /home

WORKDIR /home/project/dash_app
COPY requirements.txt /home/project/dash_app
RUN pip install --no-cache-dir -r requirements.txt

COPY docker/dashboard /home/project/dash_app