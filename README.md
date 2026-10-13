# Multi-Enterprise LLM Advisory System

An open-source, agentic Retrieval-Augmented Generation (RAG) and Fine-Tuning pipeline built for CP5403 Assessment 2. This project provides financial and operational decision support for micro-enterprises.

## 📌 System Architecture

The architecture consists of three core components modularized under the `src/` directory:

1. **RAG Retrieval Engine (`src/rag_pipeline.py`)**:
   - Built with **LangChain** and **ChromaDB**.
   - Embeds policy and financial documents using `sentence-transformers/all-MiniLM-L6-v2`.
   - Performs top-$k$ context retrieval for query grounding.

2. **LangGraph Agent Workflow (`src/agent_workflow.py`)**:
   - Implements a state-driven orchestration graph via **LangGraph**.
   - Routes queries through context retrieval (`retrieve_node`) and advisory synthesis (`advisory_generation_node`).

3. **Domain Fine-Tuning (`src/finetune.py`)**:
   - Facilitates domain adaptation for open-source foundation models on specialized enterprise advisory tasks.

---

## 🛠️ Project Structure

```text
micro-enterprise-llm-advisory/
├── requirements.txt         # Package dependencies
├── README.md                # System documentation
└── src/
    ├── finetune.py          # LLM fine-tuning logic
    ├── rag_pipeline.py      # ChromaDB vector database & retrieval
    └── agent_workflow.py    # LangGraph agent graph orchestration
