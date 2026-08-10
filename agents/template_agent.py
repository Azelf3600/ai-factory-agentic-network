"""
TEMPLATE AGENT — copy this file to build your own agent.

Shows the minimum working pattern:
  1. Read what you need from the shared state
  2. Call Gemini
  3. Return validated, structured output using a Pydantic model
  4. Append your result to the correct field in PipelineState

Run this file directly to confirm your environment + API key work
before you start writing your actual agent logic.
"""

import os
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

# Fallback for Colab Secret loading
if "GOOGLE_API_KEY" not in os.environ:
    try:
        from google.colab import userdata
        os.environ["GOOGLE_API_KEY"] = userdata.get("GOOGLE_API_KEY")
    except ImportError:
        pass

# ---- 1. Set up the LLM ----
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",  # Active model for your API key
    google_api_key=os.environ.get("GOOGLE_API_KEY"),
)


# ---- 2. Define your agent's OWN structured output shape ----
class HelloWorldOutput(BaseModel):
    company_received: str
    message: str = Field(description="A short confirmation message")


# ---- 3. The node function itself ----
def hello_world_node(state: dict) -> dict:
    company_name = state.get("company_name", "Unknown Company")

    structured_llm = llm.with_structured_output(HelloWorldOutput)
    result: HelloWorldOutput = structured_llm.invoke(
        f"Confirm you received the company name '{company_name}' "
        f"and respond with a short one-sentence greeting."
    )

    state["hello_world_result"] = result.model_dump()
    return state


# ---- 4. Minimal graph to run this node standalone ----
if __name__ == "__main__":
    graph = StateGraph(dict)
    graph.add_node("hello_world", hello_world_node)
    graph.set_entry_point("hello_world")
    graph.add_edge("hello_world", END)
    app = graph.compile()

    output = app.invoke({"company_name": "NVIDIA"})
    print("\n--- Template Agent Output ---")
    print(output)
