# Outlier Detection

A comprehensive Python library for detecting outliers and anomalies using both traditional quality control methods and modern machine learning approaches.

## Overview

This repository provides a unified pipeline for outlier detection, combining custom-implemented statistical process control methods with established machine learning algorithms.

## Algorithms

### Custom Implemented Statistical Process Control Methods
- **CUSUM (Cumulative Sum)**: Detects small shifts in the process mean by accumulating deviations from target value
- **EWMA (Exponentially Weighted Moving Average)**: Monitors process means while giving more weight to recent observations
- **T-Squared Chart**: Multivariate control chart for detecting out-of-control signals in processes with multiple correlated variables

### Integrated Machine Learning Methods
The following algorithms are integrated from established libraries and adapted to fit our pipeline:
- **Local Outlier Factor (LOF)**: Identifies samples that have a substantially lower density than their neighbors
- **Support Vector Data Description (SVDD)**: Creates a spherical boundary in feature space to separate outliers
- **Isolation Forest**: Isolates observations by randomly selecting a feature and splitting between max/min values
- **Mixture of Gaussian Density Estimation**: Models data as a mixture of several Gaussian distributions
- **PCA-Based Anomaly Detection**: Uses principal component analysis to detect outliers in high-dimensional space
- **AutoEncoder Based Anomaly Detection**: Employs deep learning to learn normal patterns and detect anomalies

## Pipeline Structure

The pipeline is designed to be modular and flexible:
```
outlier_detection/
├── src/
│   ├── statistical/          # 직접 구현한 통계적 방법들
│   │   ├── __init__.py
│   │   ├── cusum.py
│   │   ├── ewma.py
│   │   └── tsquared.py
│   │
│   ├── ml_integration/      # ML 라이브러리 통합
│   │   ├── __init__.py
│   │   ├── lof_wrapper.py
│   │   ├── iforest_wrapper.py
│   │   ├── svdd_wrapper.py
│   │   └── autoencoder_wrapper.py
│   │
│   ├── pipeline/           # 파이프라인 관련
│   │   ├── __init__.py
│   │   ├── base.py        # 기본 파이프라인 클래스
│   │   └── builder.py     # 파이프라인 구성 로직
│   │
│   └── utils/
│       ├── __init__.py
│       ├── preprocessing.py
│       ├── evaluation.py
│       └── visualization.py
│
├── requirements.txt
├── setup.py
└── README.md
```

## Implementation Details

### Custom Implementations
The statistical process control methods (CUSUM, EWMA, T-Squared) are implemented from scratch to ensure:
- Full control over algorithm parameters
- Seamless integration with our pipeline
- Optimized performance for specific use cases
- Detailed debugging capabilities

### Integrated Methods
Machine learning-based methods are integrated using established libraries (scikit-learn, PyOD) and wrapped to:
- Maintain consistent API across all methods
- Standardize input/output formats
- Provide unified parameter management
- Enable easy pipeline integration

## Usage Examples

### Univariate Analysis

```python

from outlier_detection import OutlierDetectionPipeline

from outlier_detection.statistical import CUSUM, EWMA

# For single variable monitoring

univariate_pipeline = OutlierDetectionPipeline([

    CUSUM(threshold=5),

    EWMA(lambda_=0.3)

])

# Fit and detect outliers in time series data

results = univariate_pipeline.fit_detect(time_series_data)

```

### Multivariate Analysis

```python

# For multiple variables/features

multivariate_pipeline = OutlierDetectionPipeline([

    'tsquared',  # For process monitoring

    'isolation_forest',  # For complex pattern detection

    'lof'  # For density-based detection

])

# Fit and detect outliers in multi-dimensional data

results = multivariate_pipeline.fit_detect(multivariate_data)

```

## Installation

```bash
pip install outlier-detection
```

## Requirements
- Python >= 3.7
- NumPy
- Pandas
- Scikit-learn
- PyOD
- TensorFlow (for AutoEncoder)

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.
