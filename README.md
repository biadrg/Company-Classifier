# Company-Classifier (ML Engineer Intern)

A robust company classifier for a new insurance taxonomy.

## Problem Definition

Assign (predefined) labels from the “insurance_taxonomy.csv” file to each company in the “ml_insurance_challenge.csv” file.

## Data Exploration + Cleaning

The company dataset has a shape of (9494, 5), comprising data from 9,494 companies across 5 features:
- 'description': description of company operations (9477 unique values)
- 'business_tags': categorical tags describing each business (9062 unique values)
- 'sector': industry sector (7 unique values)
- 'category': categorization (450 unique values)
- 'niche': industry niche (957 unique values)

I checked the data quality and found a few missing values (12 for ‘description’ and 27 for ‘sector’ & ‘category’), as well as 2 duplicates. Since the null values represent only a small portion of the total data amount, I have decided to replace the missing values with empty strings, so that I can still use the remaining information from the other columns still. The 2 duplicates have been dropped, to have a unique dataset.

There are only 7 unique ‘sector’ values, as shown in the 'sector_distribution.png' figure.  The most common sectors are Services and Manufacturing.

## Data Preprocessing

In order to give the model a clean input (with as little noise as can be), I joined all fields (into 'combined_fields') and removed the brankets/quotes from the ‘business_tags’ field, replaced comma separation with space separation and generally cleaned any trailing whitespaces.

## Methodology

I firstly tried to use Term Frequency–Inverse Document Frequency (TF-IDF) encoding for word appearance count and frequency. Unfortunately. TF-IDF performed reasonably as a baseline but struggled with semantic understanding. It matches keywords well but fails to grasp context. For example, a company (e.g.) that was “offering a variety of activities and events” ended up labeled as “Water Treatment Services” and so on the examples go. Its accuracy on manual labels was 30.0%

Since the keyword-based TF-IDF model failed (~ish, the model is fast and effective at matching keywords, and serves as a  baseline keyword matching approach, but the complexity of the task is higher than it can handle), I realised I needed an approach that could better "understand" the meaning of the text.

I moved on to a SentenceTransformer model (all-MiniLM-L6-v2), which converts text into meaningful vectors. This one captures meaning beyond keywords. Moreover, it handles synonyms and context. The Sentence Transformer accuracy on manual labels was 10.0%

Its performance was lower than expected, so I decided to also test a Cross Encoder (mmarco-mMiniLMv2-L12-H384-v1), which is more accurate but proved far too slow for the full dataset. 

I also tried a Sparse Encoder (SPLADE) as an alternative, which is a hybrid model combining TF-IDF and semantic classification.

## Model Evaluation

To properly evaluate the models, I manually labeled 10 companies from the dataset with ground truth labels from the insurance taxonomy. 

The models were chosen based on research, figuring out which would work best with the given dataset. It firstly started as a plan to replace each approach with a better one, but I decided to keep all approaches and have them analysed against each other instead.

Observing the graphs in the plots directory, TF-IDF and Sparse Encoder both produced scores skewed to the left, with most classifications receiving low similarity scores (0.1-0.4), which shows a general lack of confidence and a struggle to find strong matches.

Meanwhile, the Sentence Transformer had a bell-shaped distribution centered around 0.4, suggesting the model's ability to more easily distinguish between weak, medium, and strong semantic matches.

Lastly, the Cross-Encoder produced a wide range of scores, many above 0.5, which suggests its ability at identifying the single best label. This model was the most confident, considering the score distributions and model agreement.


## Bibliography

https://medium.com/@mikeyo4800/how-to-build-a-multi-label-text-classification-model-using-nlp-and-machine-learning-2e05f72aad5f

https://medium.com/@brightcode/classifying-unstructured-text-into-1800-industry-categories-with-llm-and-rag-d5fe4876841f

https://openrouter.ai/

https://github.com/UKPLab/sentence-transformers?tab=readme-ov-file#reranker-models

https://sbert.net/
