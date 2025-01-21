# Amazon Product Recommender System

## Overview

This project is a comprehensive recommender system designed to suggest Amazon products to users. It combines semantic analysis (using BERT embeddings and sentiment filtering) and collaborative filtering to deliver highly personalized recommendations.

The system is built using Python in Jupyter Notebooks and processes product descriptions and user reviews to generate recommendations.

## Features

### Content-Based Filtering:

Generates semantic embeddings using BERT for textual data (merged product descriptions and user reviews).
Applies sentiment analysis to ensure only positively-rated products are recommended.
Recommends top 10 products based on cosine similarity between user and product vectors.

### Collaborative Filtering:
Uses user reviews and ratings to suggest products based on choices of similar users.

## Workflow
The project is organized into multiple Jupyter notebooks. Below is a brief description of each notebook:

### 00_Set_Data_Structure.ipynb

Sets up the folder structure and downloads the raw Amazon dataset.
### 01_EDA_text_merged.ipynb

Performs exploratory data analysis (EDA) on the dataset.
Preprocesses and merges textual data from product descriptions and user reviews.
Exports the processed dataset for further steps.

### 02_text_analysis_BERT_embeddings_merged.ipynb

Generates BERT embeddings for the merged textual data.
Outputs embeddings for subsequent steps.
(Note: This step may take 7-8 hours on a personal computer.)

### 03_text_analysis_normalization_PCA_merged.ipynb

Applies L2 normalization to embeddings.
Reduces dimensionality using PCA to 300 dimensions for computational efficiency.

### 04_text_analysis_cos_similarity_merged.ipynb

Conducts sentiment analysis to filter out negatively-rated products.
Generates user and product vectors.
Computes cosine similarity to identify the top 300 products for each user.

### 05_text_analysis_semantic_sentiment_merged.ipynb

Generates final recommendations of the top 10 products for each user.
Filters out previously purchased products and ensures all recommendations have a positive sentiment.

### 06_collaborative_filter.ipynb

Implements collaborative filtering to recommend products based on user reviews and ratings.
Identifies user behavior patterns to suggest items liked by similar users.

## Installation and Usage
# Requirements
### Python 3.9 or higher
### Key libraries:

- for BERT embeddings
```
transformers
```

- for PCA and cosine similarity, for finding the closest aggregated product vectors for the user
    
```
scikit-learn 
```
- for data manipulation
```
pandas, numpy
```
- for text preprocessing and sentiment analysis
```
nltk, textblob 
```


## Setup
- Clone this repository:
  
```
git clone https://github.com/yourusername/amazon-recommender.git
cd amazon-recommender
```

- Install dependencies:

For Windows users:

```
pip install -r requirements.txt
```
For MAC users:
```
requirements_mac.txt
```

- Download the dataset (handled in **00_Set_Data_Structure.ipynb**).

- Run the notebooks in sequence from **00_Set_Data_Structure.ipynb** to **06_collaborative_filter.ipynb**.


## Results
Top 10 product recommendations are generated for each user based on:
- Semantic similarity (content-based filtering).
- Collaborative filtering (user behavior patterns).
Recommendations are filtered to include only positively-rated products.

## Future Work
- Enhance performance by fine-tuning BERT embeddings on the Amazon dataset.
- Incorporate a hybrid recommendation approach combining semantic and collaborative filtering.
- Optimize computation for scalability to larger datasets.













