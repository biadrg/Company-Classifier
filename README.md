# Company-Classifier (ML Engineer Intern)

A robust company classifier for a new insurance taxonomy.

## Problem Definition

Assign (predefined) labels from the “insurance_taxonomy.csv” file to each company in the “ml_insurance_challenge.csv” file.

## Data Exploration + Cleaning & Preprocessing

The company dataset has a shape of (9494, 5), so it comprises of data ('description', 'business_tags', 'sector', 'category', 'niche') from 9494 companies.

I checked the data quality and found a few missing values (12 for ‘description’ and 27 for ‘sector’ & ‘category’), as well as 2 duplicates. Since the null values represent only a small portion of the total data amount, I have decided to replace the missing values with empty strings, so that I can still use the remaining information from the other columns still. The 2 duplicates have been dropped, to have a unique dataset.

There are only 7 unique ‘sector’ values, as shown in the sector_distribution.png figure.

## Methodology

In order to give the model a clean input (with as little noise as can be), I joined all fields (into ‘combined_fields’) and removed the brankets/quotes from the ‘business_tags’ field, replaced comma separation with space separation and generally cleaned any trailing whitespaces.

I firstly tried to use Term Frequency–Inverse Document Frequency (TF-IDF) encoding for word appearance count and frequency. Unfortunately, a company (e.g.) that was “offering a variety of activities and events” ended up labeled as “Water Treatment Services” and so on the examples go.

Since the keyword-based TF-IDF model failed  (~ish, the model is fast and effective at matching keywords, and serves as a strong baseline, but the complexity of the task is higher than it can handle), I realised I needed an approach that could understand the actual meaning of the text.

I moved on to a SentenceTransformer model, which converts text into meaningful vectors. This gave me a much stronger and more logical set of classifications compared to the TF-IDF.

To be thorough, I also tested a CrossEncoder, which is more accurate but proved far too slow for the full dataset. I also tried a SparseEncoder as an alternative, which is a hybrid model combining TF-IDF and semantic classification.

The models were chosen based on research, figuring out which would work best with the given dataset. It firstly started as a plan to replace each approach with a better one, but I decided to keep all approaches and have them analysed against each other instead, generating a final consensus label


## Bibliography

https://medium.com/@mikeyo4800/how-to-build-a-multi-label-text-classification-model-using-nlp-and-machine-learning-2e05f72aad5f

https://medium.com/@brightcode/classifying-unstructured-text-into-1800-industry-categories-with-llm-and-rag-d5fe4876841f

https://openrouter.ai/

https://github.com/UKPLab/sentence-transformers?tab=readme-ov-file#reranker-models

https://sbert.net/
