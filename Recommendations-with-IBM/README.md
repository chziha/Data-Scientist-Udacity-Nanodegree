# Recommendations with IBM
This repository is for the project - Recommendations with IBM of the [Data Scientist Nanodegree by Udacity](https://www.udacity.com/course/data-scientist-nanodegree--nd025).

## Overview
In this project, the interactions that users have with articles on the IBM Watson Studio platform are analyzed to make recommendations to new/old users about new articles.

## Installation and Executing
This project requires Python 3.x and the installation of [Anaconda](https://www.anaconda.com/) or another software to execute the [Jupyter Notebook](https://ipython.org/notebook.html).
Packages required include: pandas, numpy, matplotlib, pickle, and subprocess.
     
## File Descriptions
- `data/`: contains data files from the [IBM Watson Studio platform](https://dataplatform.cloud.ibm.com/)
  - `articles_community.csv`: the dataset for the information of articles
  - `user-item-interactions.csv`: the dataset for read history by each user
- `Recommendations_with_IBM.ipynb`: main file containing the deployment of different recommendation techniques
- `Recommendations_with_IBM.html`: html verision of the main file
- `project_tests.py`: test file to validate the recommendations
- `top_n.p`: files for checking
- `user_item_matrix.p`: pre-cleaned user-by-item matrix
    
## Brief Summary
Different recommendation engines are applied including
* Rank based recommendations
* User-user based collaborative filtering
* Matrix factorization (model based) collaborative filtering

## Acknowledgements and License
Author: Zihao Chen<br/>
The data used in this project is provided by the [IBM Watson Studio platform](https://dataplatform.cloud.ibm.com/).<br/>
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

