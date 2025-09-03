"""
Data Cleaning Functions for Fake News Detection Project
"""

import pandas as pd
import re
import string


# TEXT CLEANING FUNCTIONS:

# V2:
def clean_text(text):
    """
    Comprehensive text cleaning and preprocessing for NLP tasks.
    
    Transforms raw text into a clean, normalized format suitable for machine learning
    by removing noise, standardizing format, and preserving meaningful content.
    
    Processing Pipeline:
    1.  Handles missing/empty inputs gracefully
    2.  Converts to lowercase for consistency
    3.  Removes prefixed 'b' characters (common in encoded text)
    4.  Eliminates special characters and punctuation
    5.  Removes numerical digits
    6.  Filters out single characters and very short words
    7.  Normalizes whitespace and trims edges
    
    Parameters
    ----------
    text : str, float, or None
        Raw input text to be cleaned. Can handle string values, NaN values, 
        and other non-string types that will be converted to string.
    
    Returns
    -------
    str
        Cleaned text containing only lowercase letters and single spaces.
        Returns empty string for missing or empty inputs.
    
    Examples
    --------
    >>> clean_text("SHOCKING: Trump says 2020 election was STOLEN!!!")
    'shocking trump says election was stolen'
    
    >>> clean_text("b'Scientific study confirms climate change is real'")
    'scientific study confirms climate change real'
    
    >>> clean_text("A I am going to the store at 5:30 PM.")
    'going store'
    
    >>> clean_text(None)
    ''
    
    Notes
    -----
    - Preserves words with 3+ characters to maintain semantic meaning
    - Aggressively removes noise while keeping substantive content
    - Particularly effective for news headline preprocessing
    """
    if pd.isna(text) or text == '':
        return ""
    
    # Convert to string
    text = str(text)
    
    # 1. Convert to lowercase first
    text = text.lower()
    
    # 2. Remove prefixed 'b' (from your function)
    text = re.sub(r'^b\s+', '', text)
    
    # 3. Remove special characters (BETTER APPROACH)
    # Keep letters, numbers, and basic punctuation, remove the rest
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # 4. Remove numbers (optional - from my function)
    text = re.sub(r'\d+', '', text)
    
    # 5. Remove single characters (IMPROVED VERSION)
    # Remove single letters surrounded by spaces
    text = re.sub(r'\s+[a-z]\s+', ' ', text)
    # Remove single letters at the beginning
    text = re.sub(r'^[a-z]\s+', ' ', text)
    
    # 6. Substitute multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    
    # 7. Remove very short words (from my function)
    words = text.split()
    words = [word for word in words if len(word) > 2]  # Keep words with 3+ chars
    text = ' '.join(words)
    
    # 8. Final strip
    text = text.strip()
    
    return text


# V3 Different cleaning types:

# Option 1: Gentle Cleaning (Recommended)

def gentle_clean_text(text):
    """
    Gentle text cleaning that preserves meaningful punctuation and abbreviations.
    
    Keeps: U.S., Mr., Dr., etc. (meaningful punctuation)
    Removes: Pure punctuation, excessive symbols, numbers
    
    Parameters:
    text (str): Input text to clean
    
    Returns:
    str: Cleaned text with meaningful content preserved
    """
    if pd.isna(text) or text == '':
        return ""
    
    text = str(text)
    
    # 1. Convert to lowercase
    text = text.lower()
    
    # 2. Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # 3. Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # 4. Remove standalone punctuation (but preserve meaningful abbreviations)
    # Keep: U.S., Mr., Dr., etc.
    # Remove: !, ?, ", etc.
    text = re.sub(r'\s+[^\w\s.]\s+', ' ', text)  # Punctuation with spaces
    text = re.sub(r'[^\w\s.]', ' ', text)         # Other punctuation
    
    # 5. Remove numbers (unless they're part of words)
    text = re.sub(r'\b\d+\b', '', text)  # Standalone numbers only
    
    # 6. Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text

# Option 2: Two-Level Cleaning (Best Approach)

def basic_clean_text(text):
    """
    Basic cleaning for embedding models - preserves more context.
    """
    if pd.isna(text) or text == '':
        return ""
    
    text = str(text)
    
    # Minimal cleaning for embeddings
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Remove URLs
    text = re.sub(r'<.*?>', '', text)                  # Remove HTML
    text = re.sub(r'[^\w\s.,!?]', ' ', text)           # Keep basic punctuation
    text = re.sub(r'\s+', ' ', text)                   # Normalize spaces
    text = text.strip()
    
    return text

def aggressive_clean_text(text):
    """
    Aggressive cleaning for traditional ML models (TF-IDF, etc.).
    """
    if pd.isna(text) or text == '':
        return ""
    
    text = str(text)
    
    # More aggressive cleaning
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)  # Remove ALL punctuation
    text = re.sub(r'\b\d+\b', '', text)   # Remove standalone numbers
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    # Remove very short words but keep important ones
    words = text.split()
    words = [word for word in words if len(word) > 2 or word in ['us', 'mr', 'dr']]
    text = ' '.join(words)
    
    return text

# def clean_date_column(date_series):
#     """
#     Clean and standardize date column by:
#     1. Converting to datetime where possible
#     2. Extracting year information
#     3. Creating date categories for invalid dates
    
#     Parameters:
#     date_series: pandas Series containing date values
    
#     Returns:
#     tuple: (cleaned_dates, years, date_categories)
#     """
#     from datetime import datetime
    
#     cleaned_dates = []
#     years = []
#     date_categories = []
    
#     for date_val in date_series:
#         try:
#             # Try to parse the date
#             if pd.isna(date_val) or str(date_val).strip() == '':
#                 year = 'Unknown'
#                 category = 'Unknown'
#             else:
#                 # Handle different date formats
#                 date_str = str(date_val).strip()
                
#                 # Try different date parsing strategies
#                 parsed_date = None
#                 for fmt in ['%Y-%m-%d', '%m/%d/%Y', '%Y-%m-%d %H:%M:%S', '%B %d, %Y', '%b %d, %Y']:
#                     try:
#                         parsed_date = datetime.strptime(date_str, fmt)
#                         break
#                     except ValueError:
#                         continue
                
#                 if parsed_date:
#                     year = str(parsed_date.year)
#                     # Categorize by recency
#                     if parsed_date.year >= 2020:
#                         category = 'Recent (2020+)'
#                     elif parsed_date.year >= 2015:
#                         category = 'Medium (2015-2019)'
#                     else:
#                         category = 'Old (pre-2015)'
#                 else:
#                     year = 'Unknown'
#                     category = 'Invalid Format'
            
#         except Exception as e:
#             year = 'Unknown'
#             category = 'Error'
        
#         cleaned_dates.append(date_val)  # Keep original for reference
#         years.append(year)
#         date_categories.append(category)
    
#     return cleaned_dates, years, date_categories


# MISSING DATA
def handle_missing_data(df):
    """
    Handle missing values in the dataset.
    
    Parameters:
    df (DataFrame): Input dataframe
    
    Returns:
    DataFrame: Dataframe with handled missing values
    """
    df_clean = df.copy()
    
    # Fill missing text with empty string
    text_columns = ['title', 'text']
    for col in text_columns:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna('')
    
    return df_clean


# Date cleaning / engineering:

# Add these functions to your src/data_cleaning.py file

# Data Cleaning v3:
def clean_date_column(date_series):
    """
    Robust date parsing with multiple format support and feature extraction
    
    Returns:
    - parsed_dates: datetime series
    - year: extracted year
    - quarter: extracted quarter (Q1-Q4) 
    - is_weekend: boolean for weekend dates
    """
    # Clean date strings
    date_clean = date_series.str.strip().replace('', pd.NaT)
    
    # Define date formats to try
    formats = [
        '%B %d, %Y',      # December 20, 2017
        '%d-%b-%y',       # 19-Feb-18  
        '%Y-%m-%d',       # 2017-12-20
        '%m/%d/%Y',       # 12/20/2017
        '%d/%m/%Y',       # 20/12/2017
        '%b %d, %Y',      # Dec 20, 2017
        '%Y%m%d',         # 20171220
    ]
    
    parsed_dates = pd.Series([pd.NaT] * len(date_clean), index=date_clean.index)
    
    for fmt in formats:
        try:
            temp_parsed = pd.to_datetime(date_clean, format=fmt, errors='coerce')
            mask = parsed_dates.isna() & temp_parsed.notna()
            parsed_dates[mask] = temp_parsed[mask]
        except:
            continue
    
    # Extract features
    year = parsed_dates.dt.year
    quarter = parsed_dates.dt.quarter.apply(lambda x: f'Q{x}' if pd.notna(x) else pd.NA)
    is_weekend = parsed_dates.dt.dayofweek >= 5  # 5=Saturday, 6=Sunday
    
    return parsed_dates, year, quarter, is_weekend




# Data cleaning v2
# def clean_date_column(date_series):
#     """
#     Enhanced date cleaning and feature extraction
    
#     Parameters:
#     date_series: pandas Series with date strings
    
#     Returns:
#     DataFrame with cleaned date and derived features
#     """
#     import pandas as pd
#     import numpy as np
#     from datetime import datetime
    
#     # Create a copy to avoid modifying the original
#     result_df = pd.DataFrame(index=date_series.index)
    
#     # Convert to datetime with multiple format attempts
#     date_formats = [
#         '%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%Y-%m', '%Y',
#         '%b %d, %Y', '%B %d, %Y', '%d-%b-%Y', '%d-%B-%Y'
#     ]
    
#     parsed_dates = pd.NaT * np.ones(len(date_series), dtype='datetime64[ns]')
    
#     for fmt in date_formats:
#         try:
#             mask = parsed_dates.isna()
#             if mask.any():
#                 parsed_dates[mask] = pd.to_datetime(date_series[mask], format=fmt, errors='coerce')
#         except:
#             continue
    
#     # If any dates remain unparsed, try to extract year
#     if parsed_dates.isna().any():
#         year_pattern = r'(\d{4})'
#         years = date_series[parsed_dates.isna()].str.extract(year_pattern, expand=False)
#         try:
#             parsed_dates[parsed_dates.isna()] = pd.to_datetime(years + '-01-01', errors='coerce')
#         except:
#             pass
    
#     result_df['date'] = parsed_dates
    
#     # Extract date features only from valid dates
#     valid_dates = parsed_dates.notna()
    
#     # Year
#     result_df['year'] = np.where(valid_dates, parsed_dates.dt.year, np.nan)
    
#     # Quarter (Season)
#     result_df['quarter'] = np.where(valid_dates, parsed_dates.dt.quarter, np.nan)
    
#     # Map quarter to season names
#     quarter_to_season = {1: 'Q1', 2: 'Q2', 3: 'Q3', 4: 'Q4'}
#     result_df['season'] = result_df['quarter'].map(quarter_to_season)
    
#     # Is weekend
#     result_df['is_weekend'] = np.where(valid_dates, parsed_dates.dt.dayofweek >= 5, np.nan)
    
#     # Month and day features
#     result_df['month'] = np.where(valid_dates, parsed_dates.dt.month, np.nan)
#     result_df['day_of_week'] = np.where(valid_dates, parsed_dates.dt.dayofweek, np.nan)
    
#     return result_df



def engineer_text_features(df):
    """
    Create derived text features from title and text columns
    
    Parameters:
    df: DataFrame with 'title' and 'text' columns
    
    Returns:
    DataFrame with added text features
    """
    df = df.copy()
    
    # Title features
    df['title_length'] = df['title'].str.len().fillna(0)
    df['title_word_count'] = df['title'].apply(lambda x: len(str(x).split())).fillna(0)
    
    # Text features  
    df['text_length'] = df['text'].str.len().fillna(0)
    df['text_word_count'] = df['text'].apply(lambda x: len(str(x).split())).fillna(0)
    
    # Additional text complexity features
    df['title_avg_word_length'] = np.where(
        df['title_word_count'] > 0,
        df['title_length'] / df['title_word_count'],
        0
    )
    
    df['text_avg_word_length'] = np.where(
        df['text_word_count'] > 0, 
        df['text_length'] / df['text_word_count'],
        0
    )
    
    # Flag for very short/long titles
    df['is_short_title'] = (df['title_word_count'] < 5).astype(int)
    df['is_long_title'] = (df['title_word_count'] > 15).astype(int)
    
    return df

def engineer_all_features(df):
    """
    Comprehensive feature engineering pipeline
    
    Parameters:
    df: Raw DataFrame with date, title, and text columns
    
    Returns:
    DataFrame with all engineered features
    """
    df = df.copy()
    
    # 1. Date features
    if 'date' in df.columns:
        date_features = clean_date_column(df['date'])
        df = pd.concat([df, date_features], axis=1)
    
    # 2. Text features
    if 'title' in df.columns and 'text' in df.columns:
        df = engineer_text_features(df)
    
    return df








# Pipeline stragecy:

def run_clean_pipeline(input_path, output_path, cleaning_strategy='basic'):
    """
    Run cleaning pipeline with choice of strategy.
    
    Parameters:
    cleaning_strategy: 'basic' (for embeddings) or 'aggressive' (for traditional ML)
    """
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    
    # Handle missing values
    text_columns = ['title', 'text']
    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].fillna('')
    
    # Choose cleaning strategy
    if cleaning_strategy == 'basic':
        clean_func = basic_clean_text
        print("Using basic cleaning (preserves punctuation for embeddings)")
    elif cleaning_strategy == 'aggressive':
        clean_func = aggressive_clean_text
        print("Using aggressive cleaning (for traditional ML models)")
    else:
        clean_func = clean_text
        print("Using original cleaning function")
    
    # Apply cleaning
    df['clean_title'] = df['title'].apply(clean_func)
    df['clean_text'] = df['text'].apply(clean_func)
    
    # Drop date column
    if 'date' in df.columns:
        df = df.drop(columns=['date'])
    
    print(f"Saving to {output_path}...")
    df.to_csv(output_path, index=False)
    
    print("✅ Data cleaning pipeline completed!")
    return df

# Make sure these functions are available for import
__all__ = [
    'clean_text', 
    'basic_clean_text', 
    'aggressive_clean_text', 
    'handle_missing_data',
    'clean_date_column',
    'engineer_text_features', 
    'engineer_all_features', 
    'run_clean_pipeline'
]
