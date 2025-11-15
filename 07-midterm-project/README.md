# Midterm project

To have all required packages for Midterm project you need to:
1. Install conda
2. Activate conda to dedicate the environment for specific project: conda activate base
3. Install required packages: pip install pandas numpy sklearn pickle fastapi uvicorn uv requests


Working dataset Digital Lifecycle Benchmark is free and accessible via Kaggle: https://www.kaggle.com/datasets/tarekmasryo/digital-health-and-mental-wellness
Dataset is under the Attribution 4.0 International (CC BY 4.0) license: https://creativecommons.org/licenses/by/4.0/
The license allows to share and adapt the material without restrictions except Attribution which was made above.
To have it into Midterm Project repository execute:
1. curl -L -o ./digital-health-and-mental-wellness.zip https://www.kaggle.com/api/v1/datasets/download/tarekmasryo/digital-health-and-mental-wellness
2. unzip digital-health-and-mental-wellness.zip
3. mv Data.csv digital-lifestyle.csv
Commit and push this csv-file to GitHub MLZoomcamp repository to have public link for Jupyter notebook and train.py script like this:
https://raw.githubusercontent.com/shall-it/machine-learning-zoomcamp/refs/heads/main/07-midterm-project/digital-lifestyle.csv