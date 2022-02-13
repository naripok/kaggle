FROM jupyter/tensorflow-notebook

WORKDIR /home/jovyan/work
COPY ./requirements.txt ./

RUN pip install -r requirements.txt

ENV KAGGLE_CONFIG_DIR /home/jovyan/work/.kaggle

CMD jupyter lab
