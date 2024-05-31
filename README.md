# Assessing machine learning and data imputation approaches to handle the issue of data sparsity in sports forecasting

* **Paper**: [Comming Soon]()
* **Abstract**: Sparsity is a common characteristic for datasets used in the domain of sports forecasting, mainly derived from inconsistencies in data coverage. Typically, this issue is circumvented by cutting the number of features (depth-focused) or the sample size (breadth-focused) for analysis. The present study uses an experimental approach to analyse the effects of depth- or breadth-focused analyses and data imputation to enable usage of the full sample size and feature wealth. Two forecasting models following a hybrid (i.e., a combination of classical statistical and machine learning) and a full deep learning approach are introduced to perform experiments on a dataset of more than 300,000 soccer matches. In contrast to typical soccer forecasting studies, the analysis was not restricted to one-match-ahead forecasts but used a longer forecasting horizon of up to two months ahead. Systematic differences between the two types of models were identified. The hybrid model based on a classical statistical rating models, performs strongly on depth-focused approaches while not or only marginally improving for approaches with high data breadth. The deep learning model, however, performs weakly in a depth-focused approach but profits strongly from data breadth.The improved predicting performance in cases of high data breadth suggests that a rich feature set offers better training opportunities than a less detailed set with a larger sample size. Additionally, we showcase that data imputation can be used to address data sparsity by enabling full data depth and breadth. The presented findings are relevant for advancing predictive accuracy and sports forecasting methodologies, emphasizing the viability of imputation techniques to increase data coverage in different analytical approaches.

## Table of Contents
1. [Project Structure](#project-structure)
2. [Datasets](#datasets)
3. [License](#license)
4. [Citing](#citing)
5. [Contact Information](#contact-information)

## Project Structure
This repository is structured as follow:

- [input](./input): Data samples and scripts for handling the ingestion of files
- [models](./models/): Models implementation and utilities, including each of the experiments of the paper.
- [scripts](./scripts): Tools and utilities for keeping the backend services (db, ingestion, etc.)
- [tests](./tests): Automated tests
- [utils](./utils): Utilities folder for some data wrangling, standards and computation utils.

Specifically to the contents of our paper. You can find the code to organize our splits for each of the experiments and training and validation sets [in this file](./input/_save_data_splits.py). Additionally, the Rating-based approach is documented in the models folder: [ratings](./models/rating_models/) and the [evaluation](./models/evaluation/) modules. The LSTM architecture and experiments code is documented in the [deeplearning models](./models/deepmodel/). 

## Dataset
This repository documents the solution and paper presented by the team Spoho submission for the 2023 Soccer Prediction Challenge as part of the Special issue "Machine Learning in Soccer" of the Machine Learning Journal. [:link:](https://sites.google.com/view/2023soccerpredictionchallenge/home).

The dataset is included in the documentation of the journal issue. Additionally, we used data from Transfermarkt and footballdata to enrich our data features. The full dataset used in the paper is available at request.

## License
This project is licensed under the Apache License 2.0. For more details, you can refer to the full text of the license in the LICENSE file included in this repository.

## Citing
Coming Soon

## Contact Information
If you have any questions, suggestions, or issues, please feel free to contact us:

- Dr. Fabian Wunderlich f.wunderlich@dshs-koeln.de

- Dr. Marc Garnica Caparrós m.garnica@dshs-koeln.de

- Dr. Dominik Raabe hello@raabe.ai