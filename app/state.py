# backend/app/state.py

from typing import TypedDict, List, Dict, Any
from langchain_core.runnables import Runnable

class AgentState(TypedDict):
    """
    Represents the state of our multi-agent research assistant.
    This is passed between nodes in the LangGraph.
    """
    # Add thread_id to the state to track conversations
    thread_id: str
    query: str
    original_query: str
    research_plan: str
    raw_research_results: List[Dict[str, Any]]
    summary: str
    fact_check_results: str
    messages: List[Dict[str, Any]]
    final_chain: Runnable
    final_chain_inputs: Dict[str, Any]
