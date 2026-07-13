# Orchestration Accelerator

Repo scaffold for the RAG-powered AWS Step Functions assistant described
in the proposal. This is **setup only** — folder structure, dependencies,
and empty module stubs. No Phase 1 logic has been implemented yet.

## Project layout

```
src/orchestrator/
  config.py              # stub — typed settings (env-driven)
  ingestion/
    loader.py             # stub — discovers repo files
    chunker.py             # stub — splits files into chunks
  indexing/
    gateway_client.py       # stub — OpenAI-compatible client -> internal gateway
    vectorstore.py           # stub — Chroma persistent collection
  retrieval/
    retriever.py              # stub — semantic search -> context block
  synthesis/
    planner.py                 # stub — natural language -> structured plan
    asl_generator.py            # stub — plan -> Amazon States Language JSON
    validator.py                 # stub — structural validation before deploy
  deployment/
    stepfunctions.py              # stub — boto3 create_state_machine
  cli/
    main.py                        # stub — orchestrator index|generate|deploy
tests/                              # pytest, currently just a placeholder
data/                                # local Chroma persistence (gitignored)
```

Every file above has a docstring describing what it's meant to do and a
`TODO(Phase 1)` marking what's unimplemented — nothing runs yet.

## Setup

**1. Python environment**

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

**2. Configure environment variables**

```bash
cp .env.example .env
```

Fill in `.env` — **never commit this file or paste the key anywhere
outside it**:

- `GATEWAY_API_KEY` / `GATEWAY_BASE_URL` — your internal gateway credentials.
- `GATEWAY_CHAT_MODEL` — the model alias/name your gateway exposes for chat.
- `GATEWAY_EMBEDDING_MODEL` — the model alias for embeddings.
  ⚠️ Confirm your gateway actually proxies `POST /v1/embeddings` before
  building against it — many LLM gateways only proxy `/chat/completions`.
- `STEP_FUNCTIONS_EXECUTION_ROLE_ARN` — IAM role Step Functions assumes at deploy time.
- `REPO_PATH` — path to the repository the assistant should index.

**3. AWS credentials**

Standard boto3 credential resolution applies — `AWS_PROFILE` in `.env`,
`~/.aws/credentials`, or environment variables.

**4. Verify the install**

```bash
pytest                          # placeholder test, confirms setup works
```

## Next step

Implement Phase 1 module by module, in roughly this order: `config.py` →
`ingestion/` → `indexing/` → `retrieval/` → `synthesis/` → `deployment/`
→ `cli/main.py`. Each stub's docstring says what it's responsible for.
