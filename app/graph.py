# backend/app/graph.py

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from app.state import AgentState
from app.agents import (
    generate_research_plan,
    retrieve_info,
    summarize_info,
    fact_check_summary,
    generate_final_answer,
    answer_follow_up_question, # Import the new agent
    route_initial_or_follow_up # Import the new router
)

memory = MemorySaver()

def create_research_graph():
    """
    Creates and compiles the LangGraph workflow, now with conditional routing
    for handling initial research vs. follow-up questions.
    """
    workflow = StateGraph(AgentState)

    # --- Add all nodes, including the new follow-up agent ---
    workflow.add_node("generate_research_plan", generate_research_plan)
    workflow.add_node("retrieve_info", retrieve_info)
    workflow.add_node("summarize_info", summarize_info)
    workflow.add_node("fact_check_summary", fact_check_summary)
    workflow.add_node("generate_final_answer", generate_final_answer)
    workflow.add_node("answer_follow_up_question", answer_follow_up_question)

    # --- ✨ The Entry Point is now a Conditional Router ✨ ---
    workflow.set_conditional_entry_point(
        route_initial_or_follow_up,
        {
            "generate_research_plan": "generate_research_plan",
            "answer_follow_up_question": "answer_follow_up_question",
        }
    )

    # --- Define the edges for the standard research flow ---
    workflow.add_edge("generate_research_plan", "retrieve_info")
    workflow.add_edge("retrieve_info", "summarize_info")
    workflow.add_edge("summarize_info", "fact_check_summary")
    workflow.add_edge("fact_check_summary", "generate_final_answer")
    workflow.add_edge("generate_final_answer", END)
    
    # --- The follow-up agent goes directly to the end ---
    # Its output is already prepared for streaming.
    workflow.add_edge("answer_follow_up_question", END)

    # --- Compile with Checkpointer + Interrupts ---
    app = workflow.compile(
        checkpointer=memory,
        interrupt_before=["retrieve_info"],
    )

    return app
