# AgriSmart: AI-Powered Supply Chain & Inventory Coordinator

> **"Bridging the gap between field requests and warehouse logistics with intelligent RAG and autonomous tool execution."**

## Background & Business Framing
In regional agricultural distribution, timely access to farm inputs—such as high-yield seeds, fertilizers, and equipment—is critical. Field agents, cooperative managers, and regional depots often struggle with fragmented inventory logs, opaque transit timelines, and complex safety compliance procedures. 

Delays in confirming stock or miscalculating delivery routes lead to costly stockouts during peak agricultural seasons.

## The Problem
Agricultural cooperatives maintain legacy operational files (inventory spreadsheets, warehouse logs, and shipping standard operating procedures) spread across shared drives. When field agents urgently need to verify product availability, check delivery timelines to hubs like Pokhara or Kathmandu, or look up hazardous material handling protocols, they are forced to manually sift through dense documents and disparate database sheets.

## Approach & Solution
**AgriSmart** is an end-to-end, production-ready AI assistant designed to act as a unified copilot for agricultural logistics and inventory management. 

Instead of relying on generic chat loops, the system bridges unstructured enterprise knowledge with deterministic execution:
1. **Unified Model Gateway:** Leverages **LiteLLM** to abstract model providers dynamically.
2. **Strict Structured Generation:** Uses **Pydantic and Instructor** to guarantee that every LLM response outputs validated JSON schema objects rather than ambiguous conversational text.
3. **Retrieval-Augmented Generation (RAG):** Ingests mock enterprise data (inventory CSV exports and logistics policy markdown files) into a local vector database to ground responses in real operational files.
4. **Tool-Enabled Execution:** Empowers the AI agent to execute functions that query live stock databases and compute shipping estimates.
5. **Production Wrapper:** Containerized with **Docker** and built with a lightweight **Streamlit** user interface optimized for cost-free, scalable cloud deployment.

## System Architecture
```text
[ User / Field Agent ] 
       │
       ▼ (Streamlit Web UI)
[ FastAPI / Agent Orchestrator ] 
       ├──> [ Vector Database (RAG Retrieval) ] ──> (Inventory & SOP Docs)
       ├──> [ Tool Execution Layer ] ────────────> (Live Stock & Logistics Functions)
       └──> [ LiteLLM Gateway / Instructor ] ────> (Structured JSON Output)

