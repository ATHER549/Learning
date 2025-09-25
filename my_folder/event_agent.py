import os
from datetime import datetime
from typing import List, Optional

from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field  # Updated import as per deprecation warning
from langchain.tools import tool
from langchain_openai import AzureChatOpenAI

# --- Event Details Schema ---
class EventDetails(BaseModel):
    title: str = Field(..., description="Event title")
    date: str = Field(..., description="Date in YYYY-MM-DD")
    time: str = Field(..., description="Time in HH:MM (24-hour)")
    location: Optional[str] = Field(None, description="Event location")
    attendees: Optional[List[str]] = Field(None, description="List of attendees")

# --- Tool to Add Event ---
@tool(args_schema=EventDetails)
def add_event_to_calendar(
    title: str,
    date: str,
    time: str,
    location: Optional[str] = None,
    attendees: Optional[List[str]] = None
) -> str:
    """
    Add an event to the user's calendar with the given details.
    """
    msg = f"✅ '{title}' scheduled for {date} at {time}"
    if location:
        msg += f" in {location}"
    if attendees:
        msg += f" with {', '.join(attendees)}"
    msg += ". Added to your calendar."
    print(f"[Tool]: Added '{title}' on {date} at {time}.")
    return msg

def create_event_agent():
    load_dotenv()
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Missing AZURE_OPENAI_API_KEY in .env file.")

    llm = AzureChatOpenAI(deployment_name="gpt-35-turbo")
    tools = [add_event_to_calendar]
    today = datetime.now().strftime('%Y-%m-%d (%A)')
    system_prompt = (
        f"You are a helpful calendar assistant. Today is {today}.\n"
        "Extract event details from user requests and use the add_event_to_calendar tool.\n"
        "Convert relative dates/times to absolute format. If info is missing, ask the user."
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    agent = create_openai_functions_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

if __name__ == "__main__":
    agent = create_event_agent()
    print("📅 Hi! I'm your calendar assistant.")
    print("Type your event request (or 'exit' to quit).")
    print("-" * 40)
    while True:
        user_input = input("You: ")
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye! 👋")
            break
        if user_input.strip():
            result = agent.invoke({"input": user_input})
            print("Assistant:", result["output"])
