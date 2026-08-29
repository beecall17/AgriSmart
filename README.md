# AgriSmart: AI-Powered Supply Chain & Inventory Coordinator

> Bridging the gap between field requests and warehouse logistics with Retrieval-Augmented Generation (RAG) and tool-enabled execution.

---

## What is AgriSmart

AgriSmart is an end-to-end, production-ready AI assistant and orchestration layer for agricultural logistics and inventory management. It helps field agents, cooperative managers, and depot operators quickly confirm stock, compute delivery estimates, and trigger deterministic actions (queries, lookups, and shipments) while grounding responses in enterprise data.

Key ideas:
- Ground LLM responses with a local vector DB (RAG) built from CSV exports and SOP documents.
- Enforce structured outputs using Pydantic + Instructor to produce validated JSON objects instead of free-form text.
- Execute deterministic tools (inventory queries, shipping calculations) from the agent when needed.
- Ship as a containerized app with a lightweight Streamlit UI for quick deployment.


## Features

- Unified Model Gateway (LiteLLM) to switch or configure model providers.
- Structured generation: every LLM output follows a JSON schema validated by Pydantic.
- RAG retrieval from local vector store for factual grounding.
- Tool-enabled workflows to fetch live stock, compute shipping, and update records.
- Dockerized for local development and cloud deploys; docker-compose for quick stacks.


## System Architecture

```text
[ User / Field Agent ] 
       │
       ▼ (Streamlit Web UI)
[ FastAPI / Agent Orchestrator ] 
       ├──> [ Vector Database (RAG Retrieval) ] ──> (Inventory CSVs & SOP Docs)
       ├──> [ Tool Execution Layer ] ────────────> (Live Stock & Logistics Functions)
       └──> [ LiteLLM Gateway / Instructor ] ────> (Structured JSON Output)
```


## Quickstart — Local development

Prerequisites:
- Python 3.9+ (recommended 3.10 or 3.11)
- Git
- (Optional) Docker & docker-compose for containerized runs

1. Clone the repo

   git clone https://github.com/beecall17/AgriSmart.git
   cd AgriSmart

2. Create a virtual environment and install Python dependencies

   python -m venv .venv
   source .venv/bin/activate   # macOS / Linux
   .venv\Scripts\activate     # Windows PowerShell
   pip install -r requirements.txt

3. Copy environment example and edit

   cp .env.example .env
   # Edit .env to configure any model provider keys, storage paths, or ports.

4. Prepare data

   Place your inventory CSV exports and SOP markdown files under the `data/` folder. The repository ships with example/mock data in `data/` to exercise the demo.

5. Run the app

- Run with Streamlit (dev)

  streamlit run app/main.py

- Or run with Docker Compose

  docker-compose up --build

The Streamlit UI will open at http://localhost:8501 by default (or the port configured in .env / docker-compose).


## Running in Docker

- Build locally: docker build -t agrismart:local .
- Run with docker-compose: docker-compose up

The Dockerfile and docker-compose.yml are provided to create a minimal runtime that bundles the service and its dependencies.


## Configuration

- .env.example contains environment variables used by the app (model provider keys, ports, storage paths).
- To change the vector DB path or model gateway, edit the corresponding env variables before starting the service.


## Project layout (top-level)

- app/           — Streamlit UI and FastAPI orchestrator
- data/          — Example data, inventory CSVs and SOPs used by the demo RAG pipeline
- scripts/       — helper scripts for ingestion and dataset creation
- Dockerfile
- docker-compose.yml
- requirements.txt


## Development notes

- Structured output is enforced through Pydantic models and Instructor prompts — if you add new agent actions, add matching Pydantic schemas and update the Instructor template.
- The RAG pipeline is local and uses lightweight vector stores for demo purposes; swap in a hosted vector DB for production scale.


## Contributing

Contributions and suggestions are welcome. Please open issues or PRs with clear descriptions and test coverage where appropriate.


## License

See the repository LICENSE file if present. If there is no license, consider adding one (e.g., MIT) to make reuse explicit.


---

If you'd like, I can also:
- Add a short 'Deploy to Docker Hub' guide or GitHub Actions workflow to build & publish images.
- Create example .env values and a lightweight seed dataset for quicker demos.

