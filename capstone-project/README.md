# Capstone Project - Blood cell detection for Cancer prediction

This project provides you with the ability to build and deploy Cancer prediction application related on Blood cell detection.
Cancer prediction is based on classification of the Myeloblasts cells (AML indicators): high-level risk, 12-20 micrometers, round/oval, high nuclear-cytoplasm ratio, visible nucleoli and Erythroblast cells: middle-level risk with comparing of the rest classes of Normal cells: neutrophils, monocytes and basophils. 

## Dataset details

The microscopic blood cell dataset for Cancer detection consists of high-resolution images essential for automated diagnostic systems. Each image captures detailed cellular morphology under standardized conditions, focusing on both normal and abnormal blood cells.

### Attribution
Dataset is created by author Sumith Singh Kothwal.
Dataset URL: https://www.kaggle.com/datasets/sumithsingh/blood-cell-images-for-cancer-detection

Dataset is free and accessible via Kaggle:

```bash
!curl -L -o blood-cell-images-for-cancer-detection.zip\
  https://www.kaggle.com/api/v1/datasets/download/sumithsingh/blood-cell-images-for-cancer-detection

!unzip blood-cell-images-for-cancer-detection.zip -d "blood-cell-images-for-cancer-detection"
```

### License
Dataset is under the Attribution 4.0 International (CC BY 4.0) license: https://creativecommons.org/licenses/by/4.0/
The license allows to share and adapt the material without restrictions except Attribution which was made above.

### Technical Specifications
Resolution: 1024x1024 pixels minimum
Staining: Wright-Giemsa
Magnification: 100x oil immersion (1000x total)
Color: 24-bit RGB Multiple focal planes per sample

### Quality Measures
Expert hematopathologist validation
Standardized imaging conditions
Multiple samples per cell type
Detailed preparation documentation
Complete technical metadata

### Clinical Applications
Normal vs. abnormal cell differentiation Leukemia subtype identification
Disease progression monitoring
Early detection screening
Treatment response assessment

### Image Annotations Include
Nuclear patterns and contours
Cytoplasmic features
Nucleoli presence
Cell measurements
Abnormal inclusions/Auer rods

### Machine Learning Capabilities
Automated cell classification
Quantitative feature analysis
Differential counting
Morphological abnormality detection

The dataset's structured organization and comprehensive documentation support both research initiatives and clinical applications in blood cancer diagnostics.
Its standardized format enables reliable machine learning model development for automated leukemia detection systems.

This dataset consists of 5000 images (.jpg) where the distribution is 1000 per class




## Model training
![learning_rate](./images/learning_rate.jpg)