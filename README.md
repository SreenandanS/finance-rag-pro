**Real-Time Financial News RAG Agent (Advanced Baseline on Pathway)**

**Problem Statement**

The challenge is to design a production-grade, streaming-native, agentic system for real-time financial intelligence, one that continuously processes, embeds, indexes, and reacts to incoming news data.

Traditional batch systems fall short for finance because they operate on static snapshots. This solution leverages Pathway’s streaming engine to maintain a continuously updated knowledge base of financial news and uses RAG (Retrieval-Augmented Generation) to power agentic reasoning.

The agent must:

React to new information instantly.
Maintain “always up-to-date” embeddings and sentiment metrics.
Expose live retrieval and reasoning APIs for downstream LLMs or dashboards.

**Solution Overview**
1. This project implements a dual-server, streaming-native financial RAG pipeline built entirely on Pathway with an agentic layer via LangChain + Gemini.
2. Incoming financial news (via feed.jsonl) is continuously streamed into the system, which:
3. Ingests streaming news using pw.io.jsonlines.read (live feed).
4. Cleans and splits text dynamically using TokenCountSplitter.
5. Embeds each chunk in real-time using a local SentenceTransformer model (all-MiniLM-L6-v2).
6. Indexes the embeddings inside a live DocumentStore that updates incrementally.
7. Serves a REST API for retrieval via Pathway’s built-in DocumentStoreServer.
8. Provides sentiment analysis through direct HuggingFace API calls (cardiffnlp/twitter-roberta-base-sentiment-latest).
9. Integrates with LangChain and Gemini to form a fully agentic reasoning layer with ReAct-style tool use.
10. This architecture results in a streaming, self-updating financial knowledge base that agents can query to reason over the latest market signals.

**Architecture & Pathway Usage**
<img width="605" height="536" alt="image" src="https://github.com/user-attachments/assets/2171c7ba-5996-465c-a0cf-6db6fec191ee" />


How Pathway is Used?

1. Streaming Ingestion:
Uses pw.io.jsonlines.read(..., mode="streaming", autocommit_duration_ms=500) — new news items appear within 500 ms.
2. Dynamic Transformation:
Uses Pathway’s functional API (.select(), .flatten()) to split and preprocess in-flight text data.
3. Incremental Indexing:
Real-time embeddings and cosine similarity retrieval via SentenceTransformerEmbedder + BruteForceKnnFactory.
4. Live REST Retrieval:
The DocumentStoreServer exposes /v1/retrieve, enabling LLMs and agents to pull the latest relevant articles.


Flask Extensions (Agentic & Sentiment Layer)

Pathway runs in parallel with a Flask microservice that provides higher-level intelligence:
Endpoint	Description
POST /v1/sentiment	Sentiment analysis via HuggingFace API, optional contextual retrieval
POST /v1/analyze	Retrieve + analyze sentiment jointly
POST /v1/agent	Full LangChain + Gemini ReAct Agent reasoning
GET /health	Health check and configuration status

The agent layer integrates LangChain Tools:

RetrieveNewsArticles
AnalyzeSentiment
RetrieveAndAnalyzeSentiment

and uses Google Gemini (via REST) for reasoning and orchestration.

**Tech Stack Summary**
Layer	Tool / Library
Stream Processing	Pathway
Embedding	SentenceTransformerEmbedder (all-MiniLM-L6-v2)
Indexing	BruteForceKnnFactory
Storage / API	DocumentStoreServer
Sentiment Analysis	HuggingFace Inference API (cardiffnlp/twitter-roberta-base-sentiment-latest)
Agent Framework	LangChain ReAct Agent
LLM Reasoning	Google Gemini (models/gemini-2.5-flash)
Serving Layer	Flask (Port 8091)
Streaming File	feed.jsonl
Installation & Running the Demo
Prerequisites

Docker Desktop

PowerShell or Terminal

HuggingFace API Token and (optionally) Google API Key

Step 1: Build the Docker Image
docker build -t pathway-financial-agent .

Step 2: Run the Container
docker run -p 8000:8000 -p 8091:8091 \
    -v ${PWD}/feed.jsonl:/app/feed.jsonl \
    --rm --name pathway-app pathway-financial-agent


This launches:
Pathway Document Retrieval API on :8000
Flask Sentiment & Agent API on :8091

Step 3: Try Live Demo

Run the commands in PATHWAY OUTPUT SCRIPT.pdf to:
Stream new news items into feed.jsonl
Query /v1/retrieve, /v1/sentiment, /v1/analyze, and /v1/agent
Observe live incremental updates in logs

**Future Work**
1. Integration with Ollama for fully local, offline agentic RAG.
2. Cross-modal feeds (e.g., market data + news sentiment fusion).
3. Realtime dashboards for visualization of retrieved insights.
4. Event-driven trading signal generation from agent outputs.

**Summary**
1. This project demonstrates a true real-time AI backend, where:
2. Pathway provides streaming state and indexing,
3. HuggingFace adds semantic sentiment intelligence, and
4. LangChain + Gemini turn the system into a reasoning financial agent capable of contextual understanding and adaptive response.

It’s a production-ready baseline for any real-time financial insight system — fast, reactive, and built for continuous intelligence.
