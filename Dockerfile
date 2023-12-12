# pull python base image
FROM python:3.9

# copy application files
ADD /fraud_detection_model_api /fraud_detection_model_api/

# specify working directory
WORKDIR /fraud_detection_model_api

# update pip
RUN pip install --upgrade pip

# install dependencies
RUN pip install -r requirements.txt

# expose port for application
EXPOSE 8080

# start fastapi application
CMD ["python", "app/app.py"]