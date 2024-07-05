import streamlit as st
import requests
import spacy
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
import pandas as pd
import plotly.graph_objects as go
import os
from datetime import datetime
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor

# Load environment variables
load_dotenv()

# Set page config first
st.set_page_config(page_title="Advanced Fact-Checking AI Assistant", page_icon="🔍", layout="wide")

# Download necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Load spaCy model
@st.cache_resource
def load_spacy_model():
    return spacy.load("en_core_web_sm")

nlp = load_spacy_model()

# API keys (stored in .env file)
ZENSERP_API_KEY = os.getenv("ZENSERP_API_KEY")
NEWSAPI_API_KEY = os.getenv("NEWSAPI_API_KEY")

# API URLs
ZENSERP_URL = "https://app.zenserp.com/api/v2/search"
NEWSAPI_URL = "https://newsapi.org/v2/everything"

def zenserp_search(query, api_key):
    headers = {"apikey": api_key}
    params = (("q", query), ("num", "5"), ("search_engine", "google.com"))
    response = requests.get(ZENSERP_URL, headers=headers, params=params)
    if response.status_code == 200:
        data = response.json()
        return data.get('organic', [])
    else:
        st.error(f"Error in Zenserp API call: {response.status_code}")
        return []

def newsapi_search(query, api_key):
    params = {
        "q": query,
        "apiKey": api_key,
        "pageSize": 5,
        "sortBy": "relevancy"
    }
    response = requests.get(NEWSAPI_URL, params=params)
    if response.status_code == 200:
        data = response.json()
        return data.get('articles', [])
    else:
        st.error(f"Error in NewsAPI call: {response.status_code}")
        return []

def extract_relevant_sentences(text, keywords):
    sentences = sent_tokenize(text)
    return [s for s in sentences if any(k.lower() in s.lower() for k in keywords)]

def preprocess_text(text):
    doc = nlp(text)
    return [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct]

def calculate_similarity(text1, text2):
    doc1 = nlp(text1)
    doc2 = nlp(text2)
    return doc1.similarity(doc2)

def assess_source_credibility(url):
    # This is a simplified credibility assessment
    # In a real-world scenario, you'd use a more comprehensive database or API
    credible_domains = ['bbc.com', 'nytimes.com', 'reuters.com', 'apnews.com', 'npr.org']
    return any(domain in url for domain in credible_domains)

def parse_date(date_str):
    try:
        return datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        return None

def verify_fact(fact, search_results):
    fact_tokens = preprocess_text(fact)
    verification_score = 0
    total_weight = 0
    explanations = []
    
    for result in search_results:
        title = result.get('title', '')
        snippet = result.get('snippet', result.get('description', ''))
        url = result.get('url', '')
        date_str = result.get('publishedAt', '')
        date = parse_date(date_str) if date_str else None
        
        relevant_sentences = extract_relevant_sentences(snippet, fact_tokens)
        
        for sentence in relevant_sentences:
            similarity = calculate_similarity(fact, sentence)
            credibility_score = 1.5 if assess_source_credibility(url) else 1.0
            recency_score = 1.2 if date and (datetime.now() - date).days < 30 else 1.0
            
            if similarity > 0.7:
                verification_score += similarity * credibility_score * recency_score
                total_weight += 1
                explanations.append({
                    "type": "Supporting",
                    "evidence": sentence,
                    "source": title,
                    "url": url,
                    "score": similarity,
                    "date": date_str,
                    "credibility": "High" if credibility_score > 1 else "Standard"
                })
            elif similarity > 0.5:
                verification_score += similarity * 0.5 * credibility_score * recency_score
                total_weight += 0.5
                explanations.append({
                    "type": "Partial",
                    "evidence": sentence,
                    "source": title,
                    "url": url,
                    "score": similarity,
                    "date": date_str,
                    "credibility": "High" if credibility_score > 1 else "Standard"
                })
    
    final_score = verification_score / total_weight if total_weight > 0 else 0
    
    if final_score > 0.8:
        verdict = "Likely True"
    elif final_score > 0.6:
        verdict = "Possibly True"
    elif final_score > 0.4:
        verdict = "Uncertain"
    elif final_score > 0.2:
        verdict = "Possibly False"
    else:
        verdict = "Likely False"
    
    return verdict, final_score, explanations

# Custom CSS
st.markdown("""
    <style>
    .big-font {
        font-size:20px !important;
        font-weight: bold;
    }
    .evidence-box {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🔍 Advanced Fact-Checking AI Assistant")

with st.expander("ℹ️ How to use this app", expanded=False):
    st.write("""
    1. Enter a statement you want to fact-check in the text area below.
    2. Click the 'Check Statement' button to start the fact-checking process.
    3. The app will search multiple sources and analyze the results to verify the statement.
    4. You'll see a verdict, confidence score, and supporting evidence with sources.
    5. Use the filters to sort and view the evidence as you prefer.
    """)

statement = st.text_area("Enter the statement you want to fact-check:", height=100)

col1, col2, col3 = st.columns([1,1,1])
check_button = col2.button("Check Statement", use_container_width=True)

if check_button and statement:
    with st.spinner("🕵️‍♀️ Fact-checking in progress..."):
        with ThreadPoolExecutor(max_workers=2) as executor:
            zenserp_future = executor.submit(zenserp_search, statement, ZENSERP_API_KEY)
            newsapi_future = executor.submit(newsapi_search, statement, NEWSAPI_API_KEY)
            
            zenserp_results = zenserp_future.result()
            newsapi_results = newsapi_future.result()
        
        combined_results = zenserp_results + newsapi_results
        verdict, score, explanations = verify_fact(statement, combined_results)
    
    st.markdown("## Fact-Check Results")
    st.markdown(f"**Statement:** {statement}")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"<p class='big-font'>Verdict: {verdict}</p>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<p class='big-font'>Confidence Score: {score:.2f}</p>", unsafe_allow_html=True)
    
    # Visualize the confidence score
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Confidence Score"},
        gauge = {
            'axis': {'range': [None, 1]},
            'steps': [
                {'range': [0, 0.2], 'color': "red"},
                {'range': [0.2, 0.4], 'color': "orange"},
                {'range': [0.4, 0.6], 'color': "yellow"},
                {'range': [0.6, 0.8], 'color': "lightgreen"},
                {'range': [0.8, 1], 'color': "green"}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': score}}))
    st.plotly_chart(fig)

    if explanations:
        st.markdown("## Evidence and Sources")
        
        # Create a DataFrame for easier manipulation
        df = pd.DataFrame(explanations)
        
        # Add filters
        col1, col2, col3 = st.columns(3)
        with col1:
            evidence_type = st.multiselect("Filter by evidence type:", df['type'].unique(), default=df['type'].unique())
        with col2:
            credibility = st.multiselect("Filter by source credibility:", df['credibility'].unique(), default=df['credibility'].unique())
        with col3:
            sort_by = st.selectbox("Sort by:", ["Score (High to Low)", "Score (Low to High)", "Date (Newest First)", "Date (Oldest First)"])

        # Apply filters and sorting
        filtered_df = df[df['type'].isin(evidence_type) & df['credibility'].isin(credibility)]
        if sort_by == "Score (High to Low)":
            filtered_df = filtered_df.sort_values('score', ascending=False)
        elif sort_by == "Score (Low to High)":
            filtered_df = filtered_df.sort_values('score', ascending=True)
        elif sort_by == "Date (Newest First)":
            filtered_df['date'] = pd.to_datetime(filtered_df['date'], errors='coerce')
            filtered_df = filtered_df.sort_values('date', ascending=False)
        else:
            filtered_df['date'] = pd.to_datetime(filtered_df['date'], errors='coerce')
            filtered_df = filtered_df.sort_values('date', ascending=True)

        # Display evidence
        for _, row in filtered_df.iterrows():
            with st.container():
                st.markdown(f"""
                <div class="evidence-box">
                    <p><strong>{row['type']} Evidence (Score: {row['score']:.2f}):</strong> {row['evidence']}</p>
                    <p><strong>Source:</strong> <a href="{row['url']}" target="_blank">{row['source']}</a></p>
                    <p><strong>Date:</strong> {row['date']} | <strong>Credibility:</strong> {row['credibility']}</p>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.warning("No supporting evidence found. The statement may be too specific or not widely discussed online.")

elif check_button:
    st.warning("Please enter a statement to fact-check.")

st.sidebar.header("About")
st.sidebar.info(
    "This Advanced Fact-Checking AI Assistant uses multiple APIs "
    "and advanced natural language processing to verify statements based on web search results. "
    "It considers source credibility, recency, and semantic similarity for a more accurate assessment."
)

st.sidebar.header("Disclaimer")
st.sidebar.warning(
    "This tool provides a preliminary assessment based on available online information. "
    "It should not be considered as definitive fact-checking. Always verify important information "
    "through multiple reliable sources."
)