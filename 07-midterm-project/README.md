# Midterm Project - Digital lifestyle

## Digital lifestyle app usage

Presented application has the basic level and can be used to solve the problems in the next areas: 

**Mental Health Prediction**

Build and evaluate classification models to identify individuals at elevated mental health risk based on digital behavior.

**Behavioral Analytics**

Study how technology usage influences mood, focus, and overall life balance.

**Digital Wellness Design**

Develop evidence-based interventions to encourage mindful digital habits and improve wellbeing outcomes.

The Midterm project solution focuses primarily on predicting mental health risk from digital behavior.

## Dataset for Digital lifestyle app

This dataset examines how digital lifestyles influence mental health outcomes — including anxiety, depression, stress, happiness, and productivity. It includes records from 3,500 participants across diverse backgrounds, capturing different levels of digital engagement and lifestyle balance. With 24 research-inspired features this dataset provides a robust foundation for building predictive models, correlation analyses and AI-driven wellbeing insights.

Working dataset Digital Lifecycle Benchmark is free and accessible via Kaggle: https://www.kaggle.com/datasets/tarekmasryo/digital-health-and-mental-wellness

Dataset is under the Attribution 4.0 International (CC BY 4.0) license: https://creativecommons.org/licenses/by/4.0/

The license allows to share and adapt the material without restrictions except Attribution which was made above.

To have it into Midterm Project repository execute:
```bash
curl -L -o ./digital-health-and-mental-wellness.zip https://www.kaggle.com/api/v1/datasets/download/tarekmasryo/digital-health-and-mental-wellness
unzip digital-health-and-mental-wellness.zip
mv Data.csv digital-lifestyle.csv
```

Commit and push this csv-file to GitHub MLZoomcamp repository to have public link for Jupyter notebook and train.py script like this:
https://raw.githubusercontent.com/shall-it/machine-learning-zoomcamp/refs/heads/main/07-midterm-project/digital-lifestyle.csv

### Key Feature Groups

Demographic Information
Age, Gender, Region, Income Level, Education Level

Digital Behavior
Daily Screen Time, Phone Unlocks, Notifications, Social Media Hours, and Study Time

Mental Health Indicators
Anxiety, Depression, Stress, Happiness, and Focus Scores

Risk Indicator
A High-Risk Flag summarizing digital wellbeing patterns and identifying individuals with potential mental health vulnerability.

### Dataset Design & Target Definition

This dataset was carefully modeled using patterns derived from digital wellbeing and psychology research.
Each variable was generated through realistic statistical relationships to mirror trends observed in empirical studies.

Feature Design:
Behavioral attributes (screen time, social media use, notifications, etc.) are correlated with psychological measures (stress, focus, happiness) to create meaningful multidimensional relationships.

Target Variable — high_risk_flag:
Defined through a multi-factor wellbeing score combining digital intensity and emotional indicators.
Participants with high screen time, low focus, and elevated stress/anxiety levels are classified as High Risk (1), while others are Low Risk (0).

Class Balance:
Maintains a realistic population variance, with approximately 15–20% high-risk participants, reflecting observed prevalence in mental health research.

This framework enables experimentation with machine learning models, feature interpretability, and wellbeing analytics, while maintaining alignment with real-world behavioral science findings.

### Summary Statistics

The following table summarizes the dataset's structure and intended purpose:

| Metric | Value |
|--------|-------|
| Rows | 3,500 |
| Columns | 24 |
| Target | high_risk_flag |
| Intended Use | Research · Education · Applied Machine Learning |

This dataset bridges the domains of behavioral science and artificial intelligence, offering a realistic foundation for exploring how technology use affects mental health.

## Installation

To have all required packages for Midterm project you need to:

1. Install conda
2. Activate conda to dedicate the environment for specific project: `conda activate base`
3. Install required common packages: `python -m pip install pandas numpy scikit-learn pickle fastapi uvicorn uv requests`

When you get a fresh copy of a project that already uses uv, you can install all the dependencies using the sync command: `uv sync`

## Training of model

Run Python script `train.py` based on `digital-health.ipynb` Jupyter notebook. The result is `model.bin` file.

## Local setup

Run Python script `predict.py` which using trained model `model.bin` to make predictions to start application locally:
```bash
uv run uvicorn predict:app --host 0.0.0.0 --port 9696 --reload
```

Then use Python script `risk.py` to send JSON requests and receive the response with prediction:
```bash
uv run python risk.py
```

## Dockerization

Dockerfile is applied for containerization of my local application to have an ability to deploy it everywhere with modern approach (Cloud, Kubernetes and another workloads).

Use the next commands into Midterm repository to create Docker image and get the application up and running into Docker container locally:
```bash
docker build -t predict-risk .
docker run -it --rm -p 9696:9696 predict-risk
```

With running Docker container use Python script `risk.py` as well to send JSON requests and receive the response with prediction:
```bash
uv run python risk.py
```

## Cloud deployment

Login with AWS IAM credentials like access key and secret key first.

Then execute the next commands:
```bash
aws ecr-public get-login-password --region us-east-1 | docker login --username AWS --password-stdin public.ecr.aws
aws ecr-public create-repository --repository-name predict-risk --region us-east-1
```

Tag, check and push Docker image to AWS Public ECR:
```bash
docker tag predict-risk:latest public.ecr.aws/x2b3b0k6/predict-risk:latest
docker images
docker push public.ecr.aws/x2b3b0k6/predict-risk:latest
```

Create AWS App Runner service with image specification:
```bash
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
```

Health of application can be checked just from any browser by URL: https://v2y3g8qkz5.eu-central-1.awsapprunner.com/health

URL for getting the prediction from Digital lifestyle app: https://v2y3g8qkz5.eu-central-1.awsapprunner.com/predict

Use `risk_cloud.py` Python script containing this URL to get the prediction from model deployed via AWS App Runner service:
```bash
uv run python risk_cloud.py
```