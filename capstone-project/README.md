# Capstone Project - Blood cell detection for Cancer prediction

This project provides you with the ability to build and deploy Cancer prediction application related on Blood cell detection.
Cancer prediction is based on classification of the Myeloblasts cells (AML indicators): high-level risk, 12-20 micrometers, round/oval, high nuclear-cytoplasm ratio, visible nucleoli and Erythroblast cells: middle-level risk with comparing of the rest classes of Normal cells: neutrophils, monocytes and basophils. 

## Dataset details

The microscopic blood cell dataset for Cancer detection consists of high-resolution images essential for automated diagnostic systems. Each image captures detailed cellular morphology under standardized conditions, focusing on both normal and abnormal blood cells.

### License

Dataset is uploaded under the Attribution 4.0 International (CC BY 4.0) license: https://creativecommons.org/licenses/by/4.0/
The license allows to share and adapt the material without restrictions except Attribution and Provenance section which presented below.

### Attribution and Provenance

Dataset is created by author Sumith Singh Kothwal.

Dataset URL: https://www.kaggle.com/datasets/sumithsingh/blood-cell-images-for-cancer-detection

**Sources**:

https://www.cancerimagingarchive.net/

https://www.kaggle.com/code/youssefabdelghfar/blood-cell-cancer-using-cnn-and-efficientnetb3/input


**Collection Methodology**:

The dataset merges high-quality blood cell images from The Cancer Imaging Archive (TCIA) and public datasets, specifically curated for leukemia detection research. Images were captured using Wright-Giemsa staining at 100x magnification with oil immersion, ensuring optimal visualization of cellular details. The dataset includes 5 key cell types: Basophils (with dark purple cytoplasmic granules), Erythroblasts (immature red blood cells), Monocytes (large agranulocytes with kidney-shaped nuclei), Myeloblasts (immature white blood cells indicating leukemia), and Segmented Neutrophils (mature granulocytes with segmented nuclei). All images undergo standardized preprocessing, maintaining centered cell positioning, RGB color profiles essential for morphological analysis, and consistent background normalization. This collection, distributed under CC BY-NC 4.0 license for non-commercial use, requires citation of both The Cancer Imaging Archive (TCIA) and Youssef Abdelghfar's original Kaggle dataset. It serves research, educational purposes, and development of automated leukemia detection systems.

### Dataset usage

Dataset is free and accessible via Kaggle:

```bash
!curl -L -o blood-cell-images-for-cancer-detection.zip\
  https://www.kaggle.com/api/v1/datasets/download/sumithsingh/blood-cell-images-for-cancer-detection

!unzip blood-cell-images-for-cancer-detection.zip -d "blood-cell-images-for-cancer-detection"
```

### Technical Specifications
- **Resolution**: 1024x1024 pixels minimum
- **Staining**: Wright-Giemsa
- **Magnification**: 100x oil immersion (1000x total)
- **Color**: 24-bit RGB Multiple focal planes per sample

### Quality Measures
- Expert hematopathologist validation
- Standardized imaging conditions
- Multiple samples per cell type
- Detailed preparation documentation
- Complete technical metadata

### Clinical Applications
- Normal vs. abnormal cell differentiation
- Leukemia subtype identification
- Disease progression monitoring
- Early detection screening
- Treatment response assessment

### Image Annotations Include
- Nuclear patterns and contours
- Cytoplasmic features
- Nucleoli presence
- Cell measurements
- Abnormal inclusions/Auer rods

### Machine Learning Capabilities
- Automated cell classification
- Quantitative feature analysis
- Differential counting
- Morphological abnormality detection

The dataset's structured organization and comprehensive documentation support both research initiatives and clinical applications in blood cancer diagnostics.
Its standardized format enables reliable machine learning model development for automated leukemia detection systems.

This dataset consists of 5000 images (.jpg) where the distribution is 1000 per class.
The main features of dataset are clean data and well documentation.




## Model training

Selecting of the best parameters for training process

### Selecting of learning_rate parameter:

![learning_rate](./images/learning_rate.jpg)

### Selecting of size_inner parameter:

![size_inner](./images/size_inner.jpg)

### Selecting of droprate parameter:

![droprate](./images/droprate.jpg)

### Resulting model training with the best parameters:

![result_training](./images/result_training.jpg)

## Model accuracy

Validation accuracy for the trained model reached with tuning of parameters and augmentation is more than 98%.

## Model single testing

### Testing of the model by the image from test part of dataset:

![test_image](./images/test_image.jpg)


## Docker

Check Docker service is up and running first.

Build the Docker image:
```bash
docker build -t cancer-predictor .
```

Start Docker container to test application functionality:
```bash
docker run -it --rm -p 8080:8080 cancer-predictor
```

Use this link from any browser locally to check health status of application:
```bash
http://localhost:8080/health
```

Use this link from any browser locally to enter to the Swagger UI which allows execute POST request with the example URL of test image:
```bash
http://localhost:8080/docs
```