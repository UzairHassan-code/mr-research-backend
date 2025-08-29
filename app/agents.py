# backend/app/agents.py
import os
import re
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable 
from langchain_core.output_parsers.string import StrOutputParser
from dotenv import load_dotenv
from typing import Dict, Any

from app.state import AgentState
from app.tools import get_tavily_search_tool

# --- Initialization ---
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    raise ValueError("GOOGLE_API_KEY environment variable not set.")
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0, google_api_key=google_api_key)
tavily_tool = get_tavily_search_tool()


# --- Agent Functions ---

def generate_research_plan(state: AgentState) -> Dict[str, Any]:
    """Agent: Generates a research plan."""
    print("\n---AGENT: GENERATING RESEARCH PLAN---")
    query = state["original_query"]
    
    # --- THIS IS THE FIX ---
    # The prompt now correctly uses .from_messages() to handle the system/human structure.
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert research planner. Your task is to analyze the
        user's research query and create a structured plan for investigation.
        Break down the query into specific, actionable search topics.
        Output the plan as a clear, numbered list."""),
        ("human", "User's research query: {query}")
    ])

    chain = prompt | llm | StrOutputParser()
    plan = chain.invoke({"query": query})
    return {"research_plan": plan}

def retrieve_info(state: AgentState) -> Dict[str, Any]:
    """Agent: Retrieves information based on the research plan."""
    print("\n---AGENT: RETRIEVING INFORMATION---")
    research_plan = state.get("research_plan", "")
    search_queries = re.findall(r"^\s*\d+\.\s*(.*)", research_plan, re.MULTILINE) or [research_plan]
    all_results = []
    processed_urls = set()
    for query in search_queries:
        try:
            tavily_response = tavily_tool.invoke({"query": query})
            for item in tavily_response.get('results', []):
                url, content = item.get('url'), item.get('content') or item.get('snippet')
                if url and url not in processed_urls and content:
                    all_results.append({"source": url, "content": content})
                    processed_urls.add(url)
        except Exception as e:
            print(f"Error during retrieval for query '{query}': {e}")
    return {"raw_research_results": all_results}

def summarize_info(state: AgentState) -> Dict[str, Any]:
    """Agent: Summarizes the raw research results."""
    print("\n---AGENT: SUMMARIZING INFORMATION---")
    raw_results = state.get("raw_research_results", [])
    query = state["query"]
    formatted_results = "\n\n".join([f"Source: {res['source']}\nContent: {res['content']}" for res in raw_results])
    if not formatted_results.strip():
        return {"summary": "No substantial research results were found to summarize."}
    prompt = ChatPromptTemplate.from_template(
        """You are an expert summarizer. Condense the provided research
        results into a concise, objective summary that directly addresses the
        user's original query: {query}. Highlight key findings.
        
        Retrieved Research Results:
        {results}"""
    )
    chain = prompt | llm | StrOutputParser()
    summary = chain.invoke({"query": query, "results": formatted_results})
    return {"summary": summary}

def fact_check_summary(state: AgentState) -> Dict[str, Any]:
    """Agent: Fact-checks the summary against the raw results."""
    print("\n---AGENT: FACT-CHECKING SUMMARY---")
    summary = state.get("summary", "")
    raw_results = state.get("raw_research_results", [])
    query = state["query"]
    if not raw_results:
        return {"fact_check_results": "Fact-check skipped: No original research results."}
    formatted_results = "\n\n".join([f"Source: {res['source']}\nContent: {res['content']}" for res in raw_results])
    prompt = ChatPromptTemplate.from_template(
        """You are a diligent fact-checker. Review the summary and compare it
        against the original research results. Identify any unsupported claims.
        If the summary is accurate, state 'Fact check: OK.' Otherwise, point out discrepancies.

        User's Query: {query}
        Summary to Fact-Check: {summary}
        Original Research Results:
        {results}"""
    )
    chain = prompt | llm | StrOutputParser()
    fact_check_report = chain.invoke({"query": query, "summary": summary, "results": formatted_results})
    return {"fact_check_results": fact_check_report}

def generate_final_answer(state: AgentState) -> Dict[str, Any]:
    """Agent: Prepares the final answer for streaming."""
    print("\n---AGENT: PREPARING FINAL ANSWER---")
    summary, fact_check_results, query = state["summary"], state["fact_check_results"], state["query"]
    final_prompt_template = """You are a helpful and professional research assistant.
    Synthesize the summary and fact-check report into a comprehensive, polished final answer.
    Format the response well using Markdown with headings and paragraphs.

    User's Original Query: {query}
    ---
    Summary: {summary}
    ---
    Fact Check Report: {fact_check_results}"""
    prompt = ChatPromptTemplate.from_template(final_prompt_template)
    final_chain = prompt | llm | StrOutputParser()
    return {"final_chain": final_chain, "final_chain_inputs": {"query": query, "summary": summary, "fact_check_results": fact_check_results}}


def answer_follow_up_question(state: AgentState) -> Dict[str, Any]:
    """
    Agent: Follow-up Answer Generator (Upgraded)
    Answers a follow-up question using the full context from the initial research.
    """
    print("\n---AGENT: ANSWERING FOLLOW-UP QUESTION (UPGRADED)---")
    
    summary = state.get("summary", "No previous summary available.")
    fact_check_results = state.get("fact_check_results", "No fact-check was performed.")
    raw_results = state.get("raw_research_results", [])
    
    formatted_results = "\n\n".join([f"Source: {res['source']}\nContent: {res['content']}" for res in raw_results])
    if not formatted_results.strip():
        formatted_results = "No sources were retrieved."

    follow_up_query = state.get("query", "")

    prompt_template = """You are a helpful research assistant. A user has asked a follow-up question.
Use the FULL CONTEXT of the previous research run to answer the new question.
Your context includes the summary, the fact-check report, and the raw sources.
Answer the user's question based on this existing information. Do not perform a new search.

--- PREVIOUS RESEARCH CONTEXT ---

**Summary:**
{summary}

**Fact-Check Report:**
{fact_check_results}

**Retrieved Sources:**
{formatted_results}

--- END OF CONTEXT ---

USER'S FOLLOW-UP QUESTION:
{follow_up_query}
"""
    
    prompt = ChatPromptTemplate.from_template(prompt_template)
    
    follow_up_chain = prompt | llm | StrOutputParser()
    
    return {
        "final_chain": follow_up_chain,
        "final_chain_inputs": {
            "summary": summary,
            "fact_check_results": fact_check_results,
            "formatted_results": formatted_results,
            "follow_up_query": follow_up_query
        }
    }


# --- ✨ THIS IS THE FIX: Upgraded Router Function ✨ ---
def route_initial_or_follow_up(state: AgentState) -> str:
    """
    Router: Decides whether to start a new research project, answer a follow-up,
    or regenerate the research plan.
    """
    print("\n---ROUTER: DECIDING PATH---")
    
    # Get the most recent user query
    user_query = state.get("query", "").lower()
    
    # Check if the user wants to change the plan
    # This is a simple check; a more advanced version could use an LLM call for intent detection.
    if "change" in user_query and "plan" in user_query:
        print("---DECISION: REGENERATE RESEARCH PLAN---")
        return "generate_research_plan"
        
    # Check if a summary exists from a completed research run
    if state.get("summary"):
        print("---DECISION: ANSWER FOLLOW-UP---")
        return "answer_follow_up_question"
    
    # Otherwise, it's a new research query
    else:
        print("---DECISION: INITIAL RESEARCH---")
        return "generate_research_plan"

