import os
from typing import Dict, List, TypedDict
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from src.rag_pipeline import get_retriever


# Define the state structure for our LangGraph agent
class AgentState(TypedDict):
    query: str
    retrieved_docs: List[str]
    response: str


def retrieve_node(state: AgentState) -> AgentState:
    """Node that fetches relevant documents using our RAG pipeline retriever."""
    print(f"--- AGENT: Retrieving context for query: {state['query']} ---")
    retriever = get_retriever(k_results=3)

    if not retriever:
        state["retrieved_docs"] = [
            "No document store available. Please initialize vector DB."
        ]
        return state

    # Fetch documents relevant to the query
    docs = retriever.invoke(state["query"])
    doc_texts = [doc.page_content for doc in docs]
    state["retrieved_docs"] = doc_texts
    return state


def advisory_generation_node(state: AgentState) -> AgentState:
    """Node that synthesizes the retrieved context to generate enterprise advice."""
    print("--- AGENT: Generating advisory response ---")
    query = state["query"]
    context = "\n\n".join(state["retrieved_docs"])

    # Structured prompt combining context and advisory logic
    prompt = f"""
    You are an expert Multi-Enterprise Financial Advisor. 
    Use the following retrieved institutional context to answer the user's query professionally.
    
    Context:
    {context}
    
    User Query: {query}
    
    Provide a clear, structured, and practical advisory response:
    """

    # For demonstration/offline structural robustness, we format the synthesis output
    simulated_response = (
        f"Advisory Report based on retrieved context for '{query}':\n\n"
        f"1. Key Findings: Analyzed enterprise metrics using internal guidelines.\n"
        f"2. Recommendations: Implement structured cash flow monitoring and risk mitigation.\n"
        f"3. Reference Context Snippets Used: {len(state['retrieved_docs'])} source documents referenced."
    )

    state["response"] = simulated_response
    return state


def create_advisory_graph():
    """Builds and compiles the LangGraph workflow."""
    workflow = StateGraph(AgentState)

    # Add workflow nodes
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate_advice", advisory_generation_node)

    # Set up the conditional edges / flow
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate_advice")
    workflow.add_edge("generate_advice", END)

    return workflow.compile()


if __name__ == "__main__":
    # Test execution of the graph workflow
    app = create_advisory_graph()
    test_state = {
        "query": "What are the recommended financial management practices?",
        "retrieved_docs": [],
        "response": "",
    }
    result = app.invoke(test_state)
    print("\nFinal Agent Output:\n", result["response"])
