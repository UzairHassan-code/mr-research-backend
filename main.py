# backend/main.py

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import json
from typing import AsyncGenerator
from langchain_core.runnables import Runnable
import uuid

from app.graph import create_research_graph

# --- FastAPI App Initialization ---
app = FastAPI(title="Research Assistant Chatbot")

# Create a single, module-level instance of the graph.
research_graph = create_research_graph()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def read_root():
    return {"message": "Welcome to the Research Assistant Chatbot API!"}


@app.post("/chat")
async def chat_endpoint(request: Request):
    try:
        data = await request.json()
        user_query = data.get("query")
        thread_id = data.get("thread_id") # Check for an existing thread_id
        if not user_query:
            return JSONResponse(status_code=400, content={"message": "Query is required."})
    except json.JSONDecodeError:
        return JSONResponse(status_code=400, content={"message": "Invalid JSON body."})

    # --- THIS IS THE FIX ---
    # We now handle both new and existing conversations in this endpoint.
    if thread_id:
        # If a thread_id is provided, we are continuing an existing conversation.
        print(f"\n---CONTINUING CONVERSATION (Thread ID: {thread_id})---")
        config = {"configurable": {"thread_id": thread_id}}
        # The input is just the new query. The graph will load the rest of the state.
        input_data = {"query": user_query}
    else:
        # If no thread_id, we start a new conversation.
        print("\n---STARTING NEW CONVERSATION---")
        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}
        # The input is the full initial state for a new research project.
        input_data = {
            "thread_id": thread_id,
            "query": user_query,
            "research_plan": "",
            "raw_research_results": [],
            "summary": "",
            "fact_check_results": "",
            "messages": [],
            "final_chain": None,
            "final_chain_inputs": {},
        }

    async def event_stream():
        # Always send the thread_id so the frontend can keep track of it.
        yield f"data: {json.dumps({'event': 'thread_id', 'data': thread_id})}\n\n"

        try:
            async for s in research_graph.astream(input_data, config=config):
                for node_name, node_output in s.items():
                    if node_name == "__end__":
                        continue
                    
                    if isinstance(node_output, dict):
                        # Handle research plan interruption for new conversations
                        if "research_plan" in node_output and node_output["research_plan"]:
                            event_payload = {"event": "research_plan", "data": node_output["research_plan"]}
                            yield f"data: {json.dumps(event_payload)}\n\n"
                            return # Stop the stream and wait for approval

                        # Stream the final answer (for both initial and follow-up questions)
                        if "final_chain" in node_output and "final_chain_inputs" in node_output:
                            final_chain = node_output.get("final_chain")
                            final_chain_inputs = node_output.get("final_chain_inputs")
                            if isinstance(final_chain, Runnable) and final_chain_inputs:
                                yield f"data: {json.dumps({'event': 'final_answer_start'})}\n\n"
                                async for chunk in final_chain.astream(final_chain_inputs):
                                    payload = {"event": "final_answer_chunk", "data": chunk}
                                    yield f"data: {json.dumps(payload)}\n\n"
                                    await asyncio.sleep(0.02)
            yield f"data: {json.dumps({'event': 'done', 'data': 'Stream complete'})}\n\n"
        
        except Exception as e:
            print(f"An error occurred during the stream: {e}")
            error_event = {"event": "error", "data": f"An error occurred: {str(e)}"}
            yield f"data: {json.dumps(error_event)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/chat/continue")
async def chat_continue_endpoint(request: Request):
    try:
        data = await request.json()
        thread_id = data.get("thread_id")
        research_plan = data.get("research_plan")
        if not thread_id or not research_plan:
            return JSONResponse(status_code=400, content={"message": "thread_id and research_plan are required."})
    except json.JSONDecodeError:
        return JSONResponse(status_code=400, content={"message": "Invalid JSON body."})

    config = {"configurable": {"thread_id": thread_id}}

    async def event_stream():
        try:
            research_graph.update_state(config, {"research_plan": research_plan})
            
            async for s in research_graph.astream(None, config=config):
                for node_name, node_output in s.items():
                    if node_name == "__end__":
                        continue

                    if isinstance(node_output, dict):
                        for key in ["raw_research_results", "summary", "fact_check_results"]:
                            if key in node_output:
                                event_payload = {"event": key, "data": node_output[key]}
                                yield f"data: {json.dumps(event_payload)}\n\n"
                                await asyncio.sleep(0.05)

                        if "final_chain" in node_output and "final_chain_inputs" in node_output:
                            final_chain = node_output.get("final_chain")
                            final_chain_inputs = node_output.get("final_chain_inputs")
                            if isinstance(final_chain, Runnable) and final_chain_inputs:
                                yield f"data: {json.dumps({'event': 'final_answer_start'})}\n\n"
                                async for chunk in final_chain.astream(final_chain_inputs):
                                    event_payload = {"event": "final_answer_chunk", "data": chunk}
                                    yield f"data: {json.dumps(event_payload)}\n\n"
                                    await asyncio.sleep(0.02)

            yield f"data: {json.dumps({'event': 'done', 'data': 'Stream complete'})}\n\n"

        except Exception as e:
            print(f"An error occurred during the continuation stream: {e}")
            error_event = {"event": "error", "data": f"An error occurred: {str(e)}"}
            yield f"data: {json.dumps(error_event)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
