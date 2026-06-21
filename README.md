# NLP-Based Quality Feedback Analyzer

This is an internship project focused on building an NLP-based Quality Feedback Analyzer for Amazon Product Reviews. The goal is to analyze customer reviews to extract insights regarding product quality, sentiment patterns, and representative keywords across different rating groups.

---

## Project Structure

```
.
├── app/                      # Streamlit web application (future work)
├── data/
│   ├── processed/            # Cleaned and tokenized reviews (reviews_cleaned.csv)
│   └── raw/                  # Original raw Amazon Product Reviews CSV
├── models/                   # Saved ML/DL models
│   └── lda_model/            # Serialized CountVectorizer and LDA Model pkl files
├── notebooks/                # Jupyter Notebooks for weekly deliverables
│   ├── week1_preprocessing.ipynb
│   ├── week2_sentiment.ipynb
│   └── week3_topics_entities.ipynb # Topic modeling & spaCy NER extraction
├── outputs/
│   ├── extracted_topics.csv  # Final enriched dataset with sentiment, topics, and entities
│   ├── figures/              # Saved plots, charts, and interactive visualizations
│   │   ├── lda_visualization.html        # Interactive pyLDAvis dashboard
│   │   ├── sentiment_comparison_chart.png # VADER vs DistilBERT comparison
│   │   └── topic_word_distribution.png   # Topic counts by sentiment category
│   ├── labeled_dataset.csv   # Labeled dataset with sentiment scores
│   └── wordclouds/           # Generated word cloud images
├── README.md                 # Project documentation
└── requirements.txt          # Python dependencies list
```

---

## Week 1: Preprocessing & Exploratory Analysis

In Week 1, the focus was on data ingestion, data exploration, building an NLP cleaning pipeline, and generating comparisons of high-helpfulness positive versus negative reviews.

### Tasks Completed:
1. **Data Setup**: Created directory structure and placed the raw dataset (`data/raw/Reviews.csv`) containing 568,454 product reviews.
2. **Jupyter Notebook (`notebooks/week1_preprocessing.ipynb`)**:
   - **Exploratory Data Analysis**: Loaded and analyzed columns, shapes, null counts, and rating distributions.
   - **Filtering**: Selected the 10,000 most representative unique reviews by sorting `HelpfulnessNumerator` in descending order and removing duplicate review texts.
   - **Preprocessing Pipeline**: Implemented a text cleaning function executing the exact sequence:
     `lowercase` ➔ `punctuation/special character removal` ➔ `tokenize` ➔ `NLTK stopword removal` ➔ `NLTK WordNet lemmatization`
   - **Export**: Saved both original and cleaned reviews to `data/processed/reviews_cleaned.csv`.
3. **Word Cloud Visualizations**:
   - Generated separate word clouds using the `wordcloud` library for 1-star and 5-star reviews from the cleaned text column.
   - Saved the visualizations to `outputs/wordclouds/wordcloud_1star.png` and `outputs/wordclouds/wordcloud_5star.png`.
   - Displayed both side-by-side inside the notebook.
4. **Analysis & Observations**: Included a final section comparing key terms in positive versus negative reviews.

---

## Week 2: Sentiment Analysis & Model Comparison

In Week 2, the focus was on implementing sentiment analysis using two different approaches (lexicon-based vs. transformer-based), comparing their speed and accuracy against a rating-based ground truth, and exporting a labeled dataset.

### Tasks Completed:
1. **Jupyter Notebook (`notebooks/week2_sentiment.ipynb`)**:
   - **Sampling**: Extracted a representative sample of 2,000 reviews for inference and benchmarking.
   - **Ground Truth Identification**: Derived ground-truth sentiment labels from the rating column (1-2 stars = `negative`, 3 stars = `neutral`, 4-5 stars = `positive`).
   - **VADER Sentiment**: Ran VADER (`SentimentIntensityAnalyzer`) on RAW review text, classifying compound scores into `positive`, `negative`, or `neutral`.
   - **DistilBERT Sentiment**: Loaded the pre-trained `distilbert-base-uncased-finetuned-sst-2-english` transformer pipeline and ran it on raw review text (with 400-word truncation for safety), mapping binary labels to `positive` or `negative`.
   - **Evaluation**: Benchmarked accuracy, execution speed (seconds per 1,000 reviews), and mutual agreement rate between models.
   - **Visualization**: Saved a side-by-side accuracy/speed comparison bar chart to `outputs/figures/sentiment_comparison_chart.png`.
2. **Dataset Export**: Built the final labeled dataset with columns `review text`, `rating`, `vader_label`, `distilbert_label`, `distilbert_confidence`, and `rating_based_truth`, saving it to `outputs/labeled_dataset.csv`.
3. **Comparative Observations**: Added markdown observations summarizing model performance, structural class disparities (binary vs. 3-class), and hypotheses for disagreements (sarcasm, mixed reviews, negations).

---

## Week 3: Topic Modeling & Named Entity Recognition

In Week 3, the focus is on grouping reviews into logical business categories using unsupervised topic modeling (LDA), extracting entities (products and organizations) and key complaint/feature markers using spaCy, and compiling the final merged dataset.

### Tasks Completed:
1. **Jupyter Notebook (`notebooks/week3_topics_entities.ipynb`)**:
   - **DTM Construction**: Built a Document-Term Matrix using sklearn's `CountVectorizer` on the preprocessed text tokens.
   - **LDA Topic Model**: Trained a 6-topic `LatentDirichletAllocation` model and extracted the top 10 words for each topic.
   - **Manual Label Assignment**: Mapped each topic index to a business category label:
     - Topic 0 ➔ `nutritional_profile`
     - Topic 1 ➔ `beverages_and_liquids`
     - Topic 2 ➔ `chips_and_snacks`
     - Topic 3 ➔ `order_and_delivery`
     - Topic 4 ➔ `pet_food`
     - Topic 5 ➔ `coffee_and_tea_pods`
     - Identified each review's dominant topic and added it as a `topic_label` column.
   - **Model Serialization**: Saved the trained LDA model and CountVectorizer to `models/lda_model/`.
   - **spaCy NER**: Ran spaCy's `en_core_web_sm` model on RAW review text to extract `ORG` (organization) and `PRODUCT` (product) entities.
   - **Custom Keywords Matching**: Extracted quality-specific custom keyword flags (complaints: `broken`, `damaged`, `leaked`, `stale`; positive: `fresh`, `delicious`, `organic`, `perfect`) and saved the combined list into the `entities` column.
   - **Visualizations**:
     - Generated a topic frequency distribution chart grouped by rating sentiment and saved it to `outputs/figures/topic_word_distribution.png`.
     - Created an interactive pyLDAvis HTML dashboard saved to `outputs/figures/lda_visualization.html`.
2. **Dataset Export**: Merged all columns into the final enriched format (`review text`, `rating`, `vader_label`, `distilbert_label`, `distilbert_confidence`, `topic_label`, `entities`) and saved it to `outputs/extracted_topics.csv`.

---

## Setup & Running the Notebooks

To install the necessary python packages:
```bash
pip install -r requirements.txt
python3 -m spacy download en_core_web_sm
```

To run the notebooks end-to-end:
```bash
# Run Preprocessing Notebook (Week 1)
python3 -m jupyter nbconvert --to notebook --execute --inplace notebooks/week1_preprocessing.ipynb

# Run Sentiment Analysis Notebook (Week 2)
python3 -m jupyter nbconvert --to notebook --execute --inplace notebooks/week2_sentiment.ipynb

# Run Topics & Entities Notebook (Week 3)
python3 -m jupyter nbconvert --to notebook --execute --inplace notebooks/week3_topics_entities.ipynb
```