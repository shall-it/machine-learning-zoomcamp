# Midterm project


Dataset

Working dataset Digital Lifecycle Benchmark is free and accessible via Kaggle: https://www.kaggle.com/datasets/tarekmasryo/digital-health-and-mental-wellness
Dataset is under the Attribution 4.0 International (CC BY 4.0) license: https://creativecommons.org/licenses/by/4.0/
The license allows to share and adapt the material without restrictions except Attribution which was made above.
To have it into Midterm Project repository execute:
1. curl -L -o ./digital-health-and-mental-wellness.zip https://www.kaggle.com/api/v1/datasets/download/tarekmasryo/digital-health-and-mental-wellness
2. unzip digital-health-and-mental-wellness.zip
3. mv Data.csv digital-lifestyle.csv
Commit and push this csv-file to GitHub MLZoomcamp repository to have public link for Jupyter notebook and train.py script like this:
https://raw.githubusercontent.com/shall-it/machine-learning-zoomcamp/refs/heads/main/07-midterm-project/digital-lifestyle.csv


Installation

To have all required packages for Midterm project you need to:
1. Install conda
2. Activate conda to dedicate the environment for specific project: conda activate base
3. Install required packages: pip install pandas numpy sklearn pickle fastapi uvicorn uv requests
When you get a fresh copy of a project that already uses uv, you can install all the dependencies using the sync command: uv sync


Training of model

Run Python script train.py based on digital-health.ipynb Jupyter notebook. The result is model.bin file.


Local setup

Run Python script predict.py which using trained model model.bin to make predictions to start application locally:
uv run uvicorn predict:app --host 0.0.0.0 --port 9696 --reload
Then use Python script risk.py to send JSON requests and receive the response with prediction:
uv run python risk.py


Dockerization

Dockerfile is applied for containerization of my local application to have an ability to deploy it everywhere with modern approach (Cloud, Kubernetes and another workloads)
Use the next commands into Midterm repository to create Docker image and get the application up and running into Docker container locally:
1. docker build -t predict-risk .
2. docker run -it --rm -p 9696:9696 predict-risk


Cloud deployment

aws ecr-public get-login-password --region us-east-1 | docker login --username AWS --password-stdin public.ecr.aws
aws ecr-public create-repository --repository-name predict-risk --region us-east-1

docker tag predict-risk:latest public.ecr.aws/x2b3b0k6/predict-risk:latest
docker push public.ecr.aws/x2b3b0k6/predict-risk:latest

aws apprunner create-service \
  --service-name predict-risk \
  --source-configuration '{
    "ImageRepository": {
      "ImageIdentifier": "public.ecr.aws/x2b3b0k6/predict-risk:latest",
      "ImageRepositoryType": "ECR_PUBLIC",
      "ImageConfiguration": {
        "Port": "9696"
      }
    },
    "AutoDeploymentsEnabled": false
  }' \
  --instance-configuration '{
      "Cpu": "0.25 vCPU",
      "Memory": "0.5 GB"
  }' \
  --region eu-central-1

Health of application can be checked just from any browser by URL: https://v2y3g8qkz5.eu-central-1.awsapprunner.com/health

Use risk_cloud.py script to get the prediction from model deployed via AWS App Runner service: uv run python risk_cloud.py