# app.py
import os
import logging
import pathway as pw
from typing import Optional, List, Dict, Any
import requests
from flask import Flask, request, jsonify
from threading import Thread
import json
from dotenv import load_dotenv

# load environment variables from .env if present (safe for local dev)
load_dotenv()

# LangChain imports for agentic capabilities
try:
    from langchain.agents import Tool, AgentExecutor, create_react_agent
    from langchain.prompts import PromptTemplate
    from langchain_google_genai import ChatGoogleGenerativeAI
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logging.warning("LangChain not available. Install with: pip install langchain langchain-google-genai")

# Pathway imports
try:
    from pathway.xpacks.llm.parsers import UnstructuredParser
except Exception:
    try:
        from pathway.xpacks.llm.parsers import ParseUnstructured as UnstructuredParser
        logging.warning("Using deprecated ParseUnstructured as UnstructuredParser.")
    except Exception:
        UnstructuredParser = None
        logging.warning("UnstructuredParser not available in this Pathway build.")

try:
    from pathway.xpacks.llm.splitters import TokenCountSplitter
except Exception:
    TokenCountSplitter = None
    logging.warning("TokenCountSplitter not found in pathway.xpacks.llm.splitters")

try:
    from pathway.xpacks.llm.embedders import SentenceTransformerEmbedder
except Exception:
    SentenceTransformerEmbedder = None
    logging.warning("SentenceTransformerEmbedder not present in this Pathway build.")

from pathway.stdlib.indexing import BruteForceKnnFactory
from pathway.xpacks.llm.document_store import DocumentStore
from pathway.xpacks.llm.servers import DocumentStoreServer

# ----------------------
# Configuration
# ----------------------
def clean_env_var(value):
    if value is None:
        return ""
    value = value.strip()
    if value.startswith('"') and value.endswith('"'):
        value = value[1:-1]
    elif value.startswith("'") and value.endswith("'"):
        value = value[1:-1]
    return value.strip()

DATA_FILE = clean_env_var(os.getenv("DATA_FILE", "./feed.jsonl"))
HOST = clean_env_var(os.getenv("HOST", "0.0.0.0"))
PORT = int(clean_env_var(os.getenv("PORT", "8000")))
SENTIMENT_PORT = int(clean_env_var(os.getenv("SENTIMENT_PORT", "8091")))
SENTENCE_TRANSFORMERS_MODEL = clean_env_var(os.getenv("SENTENCE_TRANSFORMERS_MODEL", "all-MiniLM-L6-v2"))
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN", "")
if HUGGINGFACE_API_TOKEN:
    HUGGINGFACE_API_TOKEN = clean_env_var(HUGGINGFACE_API_TOKEN)

# Use a sentiment classification model directly
SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"


GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
if GOOGLE_API_KEY:
    GOOGLE_API_KEY = clean_env_var(GOOGLE_API_KEY)

GEMINI_MODEL = clean_env_var(os.getenv("GEMINI_MODEL", "models/gemini-2.5-flash"))

# Configuration debug
print("=" * 50)
print("CONFIGURATION DEBUG:")
print(f"DATA_FILE: '{DATA_FILE}'")
print(f"HOST: '{HOST}'")
print(f"PORT: {PORT}")
print(f"SENTIMENT_PORT: {SENTIMENT_PORT}")
print(f"SENTIMENT_MODEL: '{SENTIMENT_MODEL}'")
print(f"GEMINI_MODEL: '{GEMINI_MODEL}'")
print(f"GOOGLE_API_KEY length: {len(GOOGLE_API_KEY)}")
print(f"HUGGINGFACE_API_TOKEN length: {len(HUGGINGFACE_API_TOKEN)}")
print(f"HUGGINGFACE_API_TOKEN (first 20): {HUGGINGFACE_API_TOKEN[:20]}...")
print("=" * 50)

# Logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("pathway-rag")

# Test the API token immediately
log.info("Testing HuggingFace API token...")
test_headers = {"Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}"}
try:
    test_response = requests.get(
        "https://huggingface.co/api/whoami-v2",
        headers=test_headers,
        timeout=5
    )
    if test_response.status_code == 200:
        log.info(f"API Token is valid! User: {test_response.json().get('name', 'Unknown')}")
    else:
        log.error(f"API Token test failed: HTTP {test_response.status_code}")
except Exception as e:
    log.error(f"API Token test error: {e}")

# ----------------------
# Ensure dependencies are available
# ----------------------
if SentenceTransformerEmbedder is None:
    raise RuntimeError(
        "SentenceTransformerEmbedder not available. "
        "Install 'sentence-transformers' and use a Pathway build with the LLM xpack.")

# ----------------------
# Read streaming JSONL input
# ----------------------
class NewsArticleSchema(pw.Schema):
    headline: str
    body: str

log.info(f"Reading JSONL data from {DATA_FILE}")
news_stream = pw.io.jsonlines.read(
    DATA_FILE,
    schema=NewsArticleSchema,
    mode="streaming",
    autocommit_duration_ms=500,
)

documents = news_stream.select(
    text=pw.apply(lambda body: body[0] if isinstance(body, tuple) else body, pw.this.body),
    headline=pw.this.headline,
)

log.info("Documents table created with columns: text, headline")

# ----------------------
# Token-based splitter
# ----------------------
chunks = documents
if TokenCountSplitter is not None:
    try:
        log.info("Attempting to use TokenCountSplitter")
        splitter = TokenCountSplitter(min_tokens=50, max_tokens=400, encoding_name="cl100k_base")
        
        chunks_with_split = chunks.select(
            chunk=splitter(pw.this.text),
            headline=pw.this.headline
        )
        
        chunks = chunks_with_split.flatten(pw.this.chunk).select(
            text=pw.apply(lambda chunk: chunk[0] if isinstance(chunk, tuple) else chunk, pw.this.chunk),
            headline=pw.this.headline
        )
        log.info("TokenCountSplitter applied successfully")
    except Exception as e:
        log.warning(f"TokenCountSplitter failed: {e}. Using full bodies as chunks.")
        chunks = documents
else:
    log.info("TokenCountSplitter not present; using full bodies as chunks")

# ----------------------
# Embedding setup
# ----------------------
embedder = SentenceTransformerEmbedder(
    model=SENTENCE_TRANSFORMERS_MODEL, 
    call_kwargs={"show_progress_bar": False}
)

try:
    emb_dim = embedder.get_embedding_dimension()
    log.info(f"Embedder dimension: {emb_dim}")
except Exception as e:
    log.warning(f"Could not detect embed dimension: {e}. Defaulting to 384.")
    emb_dim = 384

knn_factory = BruteForceKnnFactory(
    reserved_space=1000,
    embedder=embedder,
    metric=pw.engine.BruteForceKnnMetricKind.COS,
    dimensions=emb_dim,
)

# ----------------------
# Prepare documents for DocumentStore
# ----------------------
try:
    docs_for_store = chunks.select(
        data=pw.apply(lambda text: text[0] if isinstance(text, tuple) else str(text), pw.this.text),
        _metadata=pw.apply(
            lambda headline: {"headline": str(headline)},
            pw.this.headline
        )
    )
    log.info("Successfully created docs_for_store with headline metadata")
except Exception as e:
    log.warning(f"Failed to create metadata: {e}. Using data-only format.")
    docs_for_store = chunks.select(
        data=pw.apply(lambda text: text[0] if isinstance(text, tuple) else str(text), pw.this.text)
    )

# ----------------------
# Build DocumentStore
# ----------------------
document_store = DocumentStore(
    docs=docs_for_store,
    retriever_factory=knn_factory,
)

log.info("DocumentStore created successfully")

# ----------------------
# Direct HuggingFace API Sentiment Analysis
# ----------------------
log.info("=" * 50)
log.info("Setting up DIRECT HuggingFace API calls (bypassing LangChain)")

HF_API_URL = f"https://api-inference.huggingface.co/models/{SENTIMENT_MODEL}"
HF_HEADERS = {"Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}"}

def call_huggingface_api(text: str, max_retries: int = 3) -> Dict:
    """
    Call HuggingFace Inference API directly with retries.
    """
    for attempt in range(max_retries):
        try:
            response = requests.post(
                HF_API_URL,
                headers=HF_HEADERS,
                json={"inputs": text},
                timeout=30
            )
            
            if response.status_code == 200:
                return {"success": True, "data": response.json()}
            elif response.status_code == 503:
                # Model is loading
                log.warning(f"Model loading... attempt {attempt + 1}/{max_retries}")
                import time
                time.sleep(2)
                continue
            else:
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}: {response.text}"
                }
        except Exception as e:
            log.error(f"API call error (attempt {attempt + 1}): {e}")
            if attempt == max_retries - 1:
                return {"success": False, "error": str(e)}
    
    return {"success": False, "error": "Max retries exceeded"}


# Test the sentiment API
log.info(f"Testing sentiment API with model: {SENTIMENT_MODEL}")
test_result = call_huggingface_api("I love this amazing product!")
if test_result["success"]:
    log.info(f"Sentiment API test successful: {test_result['data']}")
else:
    log.error(f"Sentiment API test failed: {test_result['error']}")

log.info("=" * 50)

# ----------------------
# Sentiment Analysis Function (Direct API)
# ----------------------
def analyze_sentiment(text: str) -> str:
    """
    Analyze sentiment using HuggingFace Inference API directly.
    Returns: positive, negative, or neutral
    """
    log.info("=" * 50)
    log.info(f"SENTIMENT ANALYSIS STARTED")
    log.info(f"Input text: '{text[:100]}...'")
    
    try:
        # Call HuggingFace API
        result = call_huggingface_api(text[:512])  # Limit to 512 chars
        
        if not result["success"]:
            log.error(f"API call failed: {result['error']}")
            return f"error: {result['error']}"
        
        # Parse the response
        data = result["data"]
        log.info(f"Raw API response: {data}")
        
        # Handle different response formats
        if isinstance(data, list) and len(data) > 0:
            if isinstance(data[0], list):
                # Format: [[{'label': 'POSITIVE', 'score': 0.99}]]
                predictions = data[0]
            else:
                # Format: [{'label': 'POSITIVE', 'score': 0.99}]
                predictions = data
            
            # Find the highest scoring label
            best_prediction = max(predictions, key=lambda x: x.get('score', 0))
            label = best_prediction['label'].lower()
            score = best_prediction['score']
            
            # Map labels to our format
            if 'positive' in label or label == 'pos':
                sentiment = "positive"
            elif 'negative' in label or label == 'neg':
                sentiment = "negative"
            else:
                sentiment = "neutral"
            
            log.info(f"Sentiment: {sentiment} (confidence: {score:.2f})")
            log.info("=" * 50)
            return f"{sentiment} (confidence: {score:.2f})"
        else:
            log.error(f"Unexpected response format: {data}")
            return "error: unexpected response format"
            
    except Exception as e:
        log.error(f"SENTIMENT ANALYSIS ERROR: {type(e).__name__}: {str(e)}")
        log.exception("Full traceback:")
        log.info("=" * 50)
        return f"error: {str(e)}"


# ----------------------
# Retrieval Function
# ----------------------
def retrieve_news_articles(query: str) -> str:
    """Retrieve relevant news articles from the document store."""
    try:
        response = requests.post(
            f"http://{HOST}:{PORT}/v1/retrieve",
            json={"query": query, "k": 3},
            timeout=10
        )
        
        if response.status_code == 200:
            results = response.json()
            if not results:
                return "No relevant articles found."
            
            formatted_results = []
            for i, result in enumerate(results, 1):
                text = result.get("text", "")
                metadata = result.get("metadata", {})
                headline = metadata.get("headline", "Unknown Headline")
                score = result.get("dist", 0)
                
                formatted_results.append(
                    f"Article {i} (Relevance: {score:.3f}):\n"
                    f"Headline: {headline}\n"
                    f"Content: {text[:300]}...\n"
                )
            
            return "\n".join(formatted_results)
        else:
            return f"Error retrieving articles: HTTP {response.status_code}"
            
    except Exception as e:
        log.error(f"Retrieval error: {e}")
        return f"Error during retrieval: {str(e)}"


# ----------------------
# LangChain Agent Setup
# ----------------------
agent_executor = None

if LANGCHAIN_AVAILABLE:
    log.info("=" * 50)
    log.info("Setting up LangChain Agent")
    
    # Tool wrapper functions
    def retrieve_tool_func(query: str) -> str:
        """Retrieve relevant news articles."""
        log.info(f"[Agent Tool] Retrieving articles for: {query}")
        return retrieve_news_articles(query)
    
    def sentiment_tool_func(text: str) -> str:
        """Analyze sentiment of text."""
        log.info(f"[Agent Tool] Analyzing sentiment for: {text[:50]}...")
        return analyze_sentiment(text)
    
    def combined_analysis_tool_func(query: str) -> str:
        """Retrieve articles and analyze their sentiment."""
        log.info(f"[Agent Tool] Combined analysis for: {query}")
        articles = retrieve_news_articles(query)
        if "Error" in articles or "No relevant" in articles:
            return articles
        sentiment = analyze_sentiment(articles)
        return f"Retrieved Articles:\n{articles}\n\nOverall Sentiment: {sentiment}"
    
    # Define tools for the agent
    tools = [
        Tool(
            name="RetrieveNewsArticles",
            func=retrieve_tool_func,
            description="Useful for finding relevant news articles based on a query or topic. Input should be a search query string."
        ),
        Tool(
            name="AnalyzeSentiment",
            func=sentiment_tool_func,
            description="Useful for analyzing the sentiment (positive, negative, neutral) of text. Input should be the text to analyze."
        ),
        Tool(
            name="RetrieveAndAnalyzeSentiment",
            func=combined_analysis_tool_func,
            description="Useful for retrieving news articles about a topic AND analyzing their overall sentiment in one step. Input should be a search query."
        )
    ]
    
    # Create ReAct agent prompt
    react_prompt = PromptTemplate.from_template("""Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}""")
    
    try:
        # Initialize Google Gemini LLM for agent reasoning
        llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=GOOGLE_API_KEY,
            temperature=0.1,
            convert_system_message_to_human=True,
            transport="rest"  # Use REST instead of gRPC to avoid v1beta issues
        )
        
        # Test Gemini connection
        log.info(f"Testing Gemini API connection...")
        test_response = llm.invoke("Hello")
        log.info(f"Gemini API connection successful. Response: {test_response.content[:50]}...")
        
        # Create ReAct agent
        agent = create_react_agent(
            llm=llm,
            tools=tools,
            prompt=react_prompt
        )
        
        # Create agent executor
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            max_iterations=5,
            handle_parsing_errors=True
        )
        
        log.info("LangChain Agent initialized successfully")
        log.info(f"Agent LLM: Google Gemini - {GEMINI_MODEL}")
        log.info(f"Available tools: {[tool.name for tool in tools]}")
        
    except Exception as e:
        log.error(f"Failed to initialize LangChain Agent: {e}")
        log.exception("Full traceback:")
        agent_executor = None
    
    log.info("=" * 50)
else:
    log.warning("LangChain not available. Agent endpoint will not be functional.")


# ----------------------
# Flask API
# ----------------------
app = Flask(__name__)

@app.route('/v1/sentiment', methods=['POST'])
def sentiment_endpoint():
    """Endpoint for sentiment analysis."""
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({"error": "Missing 'text' field in request body"}), 400
        
        text = data['text']
        log.info(f"Sentiment endpoint called with text: '{text[:50]}...'")
        
        retrieve_context = data.get('retrieve_context', False)
        
        context = ""
        if retrieve_context:
            context = retrieve_news_articles(text)
            analysis_text = f"{context}\n\n{text}"
        else:
            analysis_text = text
        
        sentiment_result = analyze_sentiment(analysis_text)
        
        response = {
            "query": data['text'],
            "sentiment": sentiment_result,
        }
        
        if retrieve_context:
            response["context"] = context
        
        return jsonify(response), 200
        
    except Exception as e:
        log.error(f"Sentiment endpoint error: {e}")
        log.exception("Full traceback:")
        return jsonify({"error": str(e)}), 500


@app.route('/v1/analyze', methods=['POST'])
def analyze_endpoint():
    """Combined endpoint for retrieval + sentiment analysis."""
    try:
        data = request.get_json()
        
        if not data or 'query' not in data:
            return jsonify({"error": "Missing 'query' field in request body"}), 400
        
        query = data['query']
        k = data.get('k', 3)
        analyze_sent = data.get('analyze_sentiment', False)
        
        response_data = requests.post(
            f"http://{HOST}:{PORT}/v1/retrieve",
            json={"query": query, "k": k},
            timeout=10
        )
        
        if response_data.status_code != 200:
            return jsonify({"error": f"Retrieval failed: HTTP {response_data.status_code}"}), 500
        
        results = response_data.json()
        
        if analyze_sent and results:
            combined_text = " ".join([r.get("text", "")[:500] for r in results[:3]])
            sentiment = analyze_sentiment(combined_text)
            
            return jsonify({
                "query": query,
                "results": results,
                "sentiment_analysis": sentiment
            }), 200
        
        return jsonify({
            "query": query,
            "results": results
        }), 200
        
    except Exception as e:
        log.error(f"Analyze endpoint error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/v1/agent', methods=['POST'])
def agent_endpoint():
    """LangChain Agent endpoint for intelligent query handling."""
    if not LANGCHAIN_AVAILABLE or agent_executor is None:
        return jsonify({
            "error": "LangChain agent not available. Install langchain and langchain-community."
        }), 503
    
    try:
        data = request.get_json()
        
        if not data or 'query' not in data:
            return jsonify({"error": "Missing 'query' field in request body"}), 400
        
        query = data['query']
        log.info(f"Agent endpoint called with query: '{query}'")
        
        # Run the agent
        result = agent_executor.invoke({"input": query})
        
        return jsonify({
            "query": query,
            "response": result.get("output", "No response generated"),
            "intermediate_steps": str(result.get("intermediate_steps", []))
        }), 200
        
    except Exception as e:
        log.error(f"Agent endpoint error: {e}")
        log.exception("Full traceback:")
        return jsonify({"error": str(e)}), 500


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "service": "sentiment-analysis-api",
        "port": SENTIMENT_PORT,
        "sentiment_model": SENTIMENT_MODEL,
        "agent_available": agent_executor is not None
    }), 200


def run_flask_app():
    """Run Flask app in a separate thread."""
    log.info(f"Flask app starting on {HOST}:{SENTIMENT_PORT}")
    app.run(host=HOST, port=SENTIMENT_PORT, debug=False, threaded=True)


# ----------------------
# Start servers
# ----------------------
server = DocumentStoreServer(host=HOST, port=PORT, document_store=document_store)

log.info(f"Starting DocumentStoreServer on {HOST}:{PORT}")
try:
    server.run(threaded=True)
except TypeError:
    try:
        server.serve(threaded=True)
    except Exception as err:
        log.warning(f"server.run/serve failed: {err}")

log.info(f"Starting Flask API on {HOST}:{SENTIMENT_PORT}")
flask_thread = Thread(target=run_flask_app, daemon=True)
flask_thread.start()

log.info("Starting Pathway engine (pw.run())")
log.info(f"Available endpoints:")
log.info(f"  - POST http://{HOST}:{PORT}/v1/retrieve (Document Retrieval)")
log.info(f"  - POST http://{HOST}:{SENTIMENT_PORT}/v1/sentiment (Sentiment Analysis)")
log.info(f"  - POST http://{HOST}:{SENTIMENT_PORT}/v1/analyze (Combined Retrieval + Sentiment)")
log.info(f"  - POST http://{HOST}:{SENTIMENT_PORT}/v1/agent (LangChain Agent - NEW!)")
log.info(f"  - GET  http://{HOST}:{SENTIMENT_PORT}/health (Health Check)")
log.info(f"Using sentiment model: {SENTIMENT_MODEL}")
if agent_executor:
    log.info(f"LangChain Agent enabled with Google Gemini: {GEMINI_MODEL}")

pw.run()