
---

# **Project Plan: Fake News Classifier**

## **Table of Contents**
- [**Project Plan: Fake News Classifier**](#project-plan-fake-news-classifier)
  - [**Table of Contents**](#table-of-contents)
  - [1. Project Overview](#1-project-overview)
  - [2. NLP Cheatsheet (From Class Notes)](#2-nlp-cheatsheet-from-class-notes)
  - [3. Proposed File Structure](#3-proposed-file-structure)
  - [4. Step-by-Step Plan](#4-step-by-step-plan)
    - [Phase 1: Exploration \& Understanding (Notebook: `01_EDA.ipynb`)](#phase-1-exploration--understanding-notebook-01_edaipynb)
    - [Phase 2: Building a Processing Pipeline (File: `src/data_cleaning.py`)](#phase-2-building-a-processing-pipeline-file-srcdata_cleaningpy)
    - [Phase 3: Modeling \& Experimentation (Notebook: `02_Modeling_Experiments.ipynb`)](#phase-3-modeling--experimentation-notebook-02_modeling_experimentsipynb)
    - [Phase 4: Final Model \& Prediction (Notebook: `03_Final_Model.ipynb`)](#phase-4-final-model--prediction-notebook-03_final_modelipynb)
  - [5. How to Approach Different Models \& Features](#5-how-to-approach-different-models--features)
  - [6. Phase 5: Showcasing the Model (Optional "Nice-to-Have")](#6-phase-5-showcasing-the-model-optional-nice-to-have)
    - [App Concept: "The Fake News Detective"](#app-concept-the-fake-news-detective)
    - [Planned Features \& Mock-Up:](#planned-features--mock-up)
    - [Implementation Plan:](#implementation-plan)
    - [Why This is a Great Idea:](#why-this-is-a-great-idea)
    - [Updated File Structure (Final):](#updated-file-structure-final)
  - [🎯 **Model Testing Strategy**](#-model-testing-strategy)
    - [**Approach 1: Embedding Encoding**](#approach-1-embedding-encoding)
    - [**Approach 2: Traditional NLP Pipeline**](#approach-2-traditional-nlp-pipeline)
    - [**Approach 3: Advanced (Transformer Models)**](#approach-3-advanced-transformer-models)
  - [📁 **File Structure \& Naming**](#-file-structure--naming)

---

## 1. Project Overview

**Goal:** Build a machine learning model to classify news headlines as `Fake (0)` or `Real (1)`.

**Final Deliverable:** A predicted label for each entry in `dataset/validation_data.csv`.

**Key Principle:** *Modularity*. We will break the project into independent, reusable parts. This makes testing different ideas much easier.

---

## 2. NLP Cheatsheet (From Class Notes)

| Concept | What it is | Why we use it | Key Python Tool |
| :--- | :--- | :--- | :--- |
| **Tokenization** | Splitting text into words or sub-words. | Computers need to work with individual units. | `nltk.word_tokenize()` |
| **Stopword Removal** | Removing very common words (e.g., "the", "and", "is"). | They add noise, not meaning. Helps focus on important words. | `sklearn.feature_extraction.text.ENGLISH_STOP_WORDS` |
| **TF-IDF Vectorization** | Converts text into numbers. Weighs words by how important they are to a document in a collection. | Best way to turn text into features for many models. | `sklearn.feature_extraction.text.TfidfVectorizer` |
| **n-grams** | Groups of `n` consecutive words (e.g., "not good" is a 2-gram/bigram). | Captures phrases and context, not just single words. | `TfidfVectorizer(ngram_range=(1, 2))` |
| **Train/Test Split** | Dividing data into a set for training and a hidden set for testing. | To honestly evaluate how the model performs on new, unseen data. | `sklearn.model_selection.train_test_split` |
| **Classification Model** | An algorithm that learns patterns from features (X) to predict a category (y). | To make the actual prediction. | `LogisticRegression`, `RandomForest` |
| **Model Evaluation** | Measuring how good the predictions are. | To choose the best model and know how well it will work in the real world. | `accuracy_score`, `classification_report`, `roc_auc_score`, `roc_curve`, `confusion_matrix` |

---

## 3. Proposed File Structure

```
your_project_folder/
│
├── dataset/
│   ├── 00_raw/                   # NEVER change files here manually!
│   │   ├── data.csv              # Original, untouched data
│   │   └── validation_data.csv   # Original, untouched validation data
│   │
│   ├── 01_interim/               # For cleaned/processed data
│   │   ├── cleaned_data.csv      # Output from data_cleaning.py
│   │   └── cleaned_validation.csv
│   │
│   ├── 02_primary/              # For feature-engineered data (optional)
│   │   ├── model_training_data.csv
│   │   └── model_validation_data.csv
│   │
│   └── 03_processed/            # Final dataset used for modeling
│       └── (Often not needed, as we use arrays in notebooks)│
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Modeling_Experiments.ipynb
│   ├── 03_Model_Testing_Embeddings.ipynb          ← NEW
│   ├── 04_Model_Testing_Traditional_NLP.ipynb     ← NEW  
│   └── 05_Model_Testing_Advanced.ipynb            ← NEW (optional)
│   └── 100_Final_Model.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_cleaning.py    # Preprocessing functions
│   └── feature_engineering.py # TF-IDF, n-grams code
│
├── models/                   
│   ├── best_model.pkl
│   └── tfidf_vectorizer.pkl
│
├── app.py                    # The Streamlit app
│
├── validation_predictions.csv  # FINAL OUTPUT
│
└── project-plan.md             # This file!
```


---

## 4. Step-by-Step Plan

### Phase 1: Exploration & Understanding (Notebook: `01_EDA.ipynb`)
**Goal:** Understand the data deeply without touching any models.

**Tasks:**
1.  Load `data.csv`.
2.  **Basic Info:** Check shape, data types, missing values.
3.  **Target Analysis:** How many fake vs. real news? (Bar chart). Is the dataset balanced?
4.  **Text Length Analysis:** Is there a difference in the length of fake vs. real headlines? (Boxplot).
5.  **Word Frequency:** What are the most common words in Fake news? In Real news? (Word clouds or bar charts).
6.  **Subject & Date:** Analyze the `subject` and `date` columns. Do certain subjects have more fake news?

**Outcome:** A document with insights that will guide preprocessing and modeling (e.g., "Real news headlines are longer on average.").

### Phase 2: Building a Processing Pipeline (File: `src/data_cleaning.py`)
**Goal:** Create reusable functions to clean and prepare text. This is the most important step!

**Functions to Create:**
- `clean_text(text)`: A function that takes a raw headline and returns a cleaned one.
    - Convert to lowercase.
    - Remove punctuation, numbers, extra spaces.
    - (Maybe) remove stopwords.
    - (Maybe) perform stemming/lemmatization.
- `save_clean_data()`: Runs `clean_text()` on the entire dataset and saves a cleaned version to a new file. This keeps our raw data safe.

**Why a `.py` file?** So we can `import` these same functions in *every* notebook, ensuring we clean our data the same way every time.

### Phase 3: Modeling & Experimentation (Notebook: `02_Modeling_Experiments.ipynb`)
**Goal:** Test different ideas to see what works best. This is our "playground".

**Strategy: Use a "Grid of Experiments" Approach**

We will systematically try different combinations of **features** and **models**.

| Experiment ID | Features (X) | Model | Notes | Test Accuracy |
| :--- | :--- | :--- | :--- | :--- |
| **A1** | `title` (cleaned) | `LogisticRegression` | Baseline | |
| **A2** | `title` (cleaned) | `RandomForest` | Compare to A1 | |
| **B1** | `text` (first 100 chars) | `LogisticRegression` | Does full text help? | |
| **C1** | `title` + bigrams | `LogisticRegression` | Does context help? | |
| **C2** | `title` + bigrams | `RandomForest` | | |
| **D1** | `title` + `subject` (as a category) | `LogisticRegression` | Does topic help? | |

| **XX** | `Add other Models from the research step` (as a category) | `ExampleClassifier` | Does topic help? | |


> Add other Models from the research `step 2`:
>> - ADA Boster
>> - X Boster
>> - etc. `(fill in)`

>>> Add more models we saw in class.

<br>


**How to manage this?**
1.  **Import** your cleaning functions from `src.data_cleaning import clean_text`.
2.  For each experiment:
    -   Prepare the specific feature (e.g., `X = df['cleaned_title']` for A1, or `X = df[['cleaned_title', 'subject_encoded']]` for D1).
    -   Apply TF-IDF if it's text.
    -   Split into train/test.
    -   Train the model.
    -   **Evaluate on the test set and record the accuracy.**
3.  **Critical:** Don't touch the validation data (`validation_data.csv`) yet! We only use our own test set to choose the best model.

**Outcome:** A table showing which combination (features + model) gave the best score on the test set.

### Phase 4: Final Model & Prediction (Notebook: `03_Final_Model.ipynb`)
**Goal:** Train the best model on *all* our training data and predict on the true validation set.

**Tasks:**
1.  **Identify the Winner:** From Phase 3, choose the best experiment (e.g., `C1: LogisticRegression` on `title + bigrams`).
2.  **Prepare the Final Data:** Combine your training and test sets from `data.csv` to create one large training dataset. (More data = better model).
3.  **Train the Final Model:** Preprocess this full dataset and train your chosen model on it.
4.  **Predict:** Load `validation_data.csv`, clean it using your `clean_text()` function, transform it with the fitted TF-IDF, and make predictions.
5.  **Save:** Save the predictions to `validation_predictions.csv`.

**Outcome:** Your final deliverable file.

---

## 5. How to Approach Different Models & Features

Your teacher's question is excellent. Here is the simple strategy:

1.  **Start Simple:** Your first model (e.g., Logistic Regression on just cleaned titles) is your **baseline**. Every other experiment must beat this baseline to be considered an improvement.
2.  **Change One Thing at a Time:** Don't change the model and the features simultaneously. In the experiment grid above:
    -   **A1 vs. A2** tests different models on the *same feature*.
    -   **A1 vs. C1** tests different features on the *same model*.
    This tells you exactly what caused any improvement.
3.  **Features are Often More Important than the Model:** It's very common that using better features (e.g., adding bigrams) improves accuracy more than switching from a good model to a more complex one. This is why we experiment.
4.  **Keep a Log:** Use a simple table (like above) or a text cell in your notebook to record every experiment and its result. This is your "lab notebook" and is incredibly valuable.

This plan makes you a scientist, not just a coder. You form hypotheses (e.g., "Using bigrams will improve accuracy"), run experiments to test them, and analyze the results.

---

## 6. Phase 5: Showcasing the Model (Optional "Nice-to-Have")

**Goal:** Create an interactive web application that allows non-technical users to see your fake news detector in action.

**Tool:** **Streamlit** - A Python library that makes it incredibly easy to turn data scripts into shareable web apps.

### App Concept: "The Fake News Detective"

A simple, intuitive app where a user can paste any news headline and get an instant prediction.

### Planned Features & Mock-Up:

1.  **Title & Description:**
    *   "🔍 The Fake News Detective"
    *   Brief description: "This tool uses a machine learning model to analyze news headlines and predict if they are more likely to be real or fake. Paste a headline below to try it out!"

2.  **User Input Box:**
    *   A large text area with a placeholder: "Paste a news headline here... (e.g., 'Scientists confirm that coffee is good for your health')"

3.  **"Analyze Headline" Button:**
    *   The user clicks this to run the model.

4.  **Results Display:**
    *   **Clear Visual Indicator:**
        *   If **Fake**: A red card/section with text: "⚠️ Our analysis suggests this headline is likely **FAKE**."
        *   If **Real**: A green card/section with text: "✅ Our analysis suggests this headline is likely **REAL**."
    *   **Confidence Score:** A progress bar or percentage showing the model's confidence in its prediction (e.g., "87% confident").
    *   **Explanation (Key Feature):** A section called "Why?" that shows the top 3-5 words that most influenced the decision.
        *   *Example: "The words 'SHOCKING', 'ALIENS', and 'SECRET' are often found in fake news."*
        *   This makes the "black box" model transparent and educational.

### Implementation Plan:

1.  **Create a new file:** `app.py` in the main project folder.
2.  **Save your best model:** Use `joblib` or `pickle` to save your trained model (`model.pkl`) and TF-IDF vectorizer (`vectorizer.pkl`) from **Phase 4**.
3.  **Build the app:**
    *   Import `streamlit`, `joblib`, `numpy`.
    *   Load the pre-saved model and vectorizer.
    *   Use `st.title()`, `st.text_input()`, and `st.button()` to create the layout.
    *   When the button is clicked, clean the input text, transform it with the loaded vectorizer, and get a prediction/probability from the model.
    *   Use `st.progress()`, `st.success()`, and `st.error()` to display the results nicely.
4.  **Run it locally:** Test the app by running `streamlit run app.py` in your terminal.

### Why This is a Great Idea:

*   **Portfolio Gold:** It transforms your project from a technical script into a tangible product.
*   **Demonstrates Communication Skills:** It shows you can explain complex ML concepts in a simple way.
*   **Proves End-to-End Understanding:** It covers the full pipeline: from data, to training, to deployment in a user-friendly interface.


-----

### Updated File Structure (Final):


Excellent! Now we're moving to the most exciting part - model testing. Let's create a structured approach for your modeling experiments.

## 🎯 **Model Testing Strategy**

### **Approach 1: Embedding Encoding**
- **Use:** `basic_clean_text` (preserves punctuation, context)
- **Models:** Word2Vec, GloVe, or pre-trained embeddings (like spaCy)
- **Best for:** Deep Learning, Neural Networks

### **Approach 2: Traditional NLP Pipeline**
- **Use:** `aggressive_clean_text` + Lemmatization/Stemming
- **Models:** TF-IDF + Logistic Regression/Random Forest
- **Best for:** Traditional Machine Learning

### **Approach 3: Advanced (Transformer Models)**
- **Use:** Minimal cleaning (just lowercase)
- **Models:** BERT, DistilBERT (Hugging Face)
- **Best for:** State-of-the-art performance

---

## 📁 **File Structure & Naming**

```
your_project_folder/
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Data_Cleaning.ipynb
│   ├── 03_Model_Testing_Embeddings.ipynb          ← NEW
│   ├── 04_Model_Testing_Traditional_NLP.ipynb     ← NEW  
│   └── 05_Model_Testing_Advanced.ipynb            ← NEW (optional)
│
├── models/                                        ← For saving trained models
│   └── (will contain saved models)
│
└── src/
    ├── __init__.py
    ├── data_cleaning.py
    └── modeling_utils.py                         ← NEW (helper functions)
```
