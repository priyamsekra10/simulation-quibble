"""Conversation Simulator API module for simulating AI and user conversations."""
import logging
import os
import uuid
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
import openai

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

app = FastAPI(title="Conversation Simulator API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class SimulationRequest(BaseModel):
    """Request model for conversation simulation."""
    system_prompt: str
    simulated_user_prompt: str
    max_messages: int = 5
    system_model: str = "gpt-4o-mini"
    simulated_user_model: str = "gpt-4o-mini"
    user_id: Optional[str] = None

class MessageResponse(BaseModel):
    """Model for a single message in the conversation."""
    role: str
    content: str

class SimulationResponse(BaseModel):
    """Response model for conversation simulation results."""
    request_id: str
    user_id: Optional[str] = None
    conversation: List[MessageResponse]

def convert_message_to_dict(message):
    """Convert LangChain message to OpenAI format with proper role."""
    if isinstance(message, HumanMessage):
        return {"role": "user", "content": message.content}
    return {"role": "assistant", "content": message.content}

def my_chat_bot(messages: List, system_prompt: str, system_model: str) -> dict:
    """Generate a response from the chat bot using the specified model and prompt."""
    # Convert LangChain messages to OpenAI format with proper roles
    formatted_messages = [{"role": "system", "content": system_prompt}]
    
    for msg in messages:
        formatted_messages.append(convert_message_to_dict(msg))
    
    completion = openai.chat.completions.create(
        messages=formatted_messages, model=system_model
    )
    return completion.choices[0].message.model_dump()

def create_simulated_user(instructions: str, model_name: str):
    """Create a simulated user with given instructions and model."""
    system_prompt_template = """{instructions}
    When you are finished with your initial objective, respond with 'FINISHED' to end the chat."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt_template),
        MessagesPlaceholder(variable_name="messages"),
    ])
    return prompt.partial(instructions=instructions) | ChatOpenAI(model=model_name)

def _swap_roles(messages):
    """Swap roles between AI and Human messages."""
    return [
        HumanMessage(content=m.content) if isinstance(m, AIMessage)
        else AIMessage(content=m.content) for m in messages
    ]

@app.post("/simulate", response_model=SimulationResponse)
async def simulate_conversation(request: SimulationRequest):
    """
    Simulate a conversation between an AI assistant and a user.
    
    Returns a complete conversation history.
    """
    if not os.environ.get("OPENAI_API_KEY"):
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set in environment")
    
    request_id = str(uuid.uuid4())
    user_id = request.user_id or str(uuid.uuid4())
    
    logging.info("Starting new simulation: %s for user: %s", request_id, user_id)
    logging.info("System prompt: %s", request.system_prompt)
    logging.info("Simulated user prompt: %s", request.simulated_user_prompt)
    logging.info("System model: %s", request.system_model)
    logging.info("Simulated user model: %s", request.simulated_user_model)
    
    try:
        simulated_user = create_simulated_user(
            request.simulated_user_prompt, request.simulated_user_model
        )
        messages = []
        conversation_history = []
        
        # Start the conversation with a user message
        user_message = HumanMessage(content="Hi! I need some help.")
        messages.append(user_message)
        conversation_history.append({"role": "User", "content": user_message.content})
        
        for _ in range(request.max_messages):
            # Get AI response
            chat_bot_response = my_chat_bot(
                messages, request.system_prompt, request.system_model
            )
            ai_message = AIMessage(content=chat_bot_response["content"])
            conversation_history.append({"role": "AI", "content": ai_message.content})
            
            if "FINISHED" in ai_message.content:
                break
            
            messages.append(ai_message)
            
            # Get user response
            swapped_messages = _swap_roles(messages)
            user_response = simulated_user.invoke({"messages": swapped_messages})
            user_message = HumanMessage(content=user_response.content)
            conversation_history.append({"role": "User", "content": user_message.content})
            
            if "FINISHED" in user_message.content:
                break
            
            messages.append(user_message)
        
        logging.info("Simulation %s completed with %d messages for user %s", 
                    request_id, len(conversation_history), user_id)
        
        return SimulationResponse(
            request_id=request_id,
            user_id=user_id,
            conversation=conversation_history
        )
    except Exception as e:
        logging.error("Simulation %s error for user %s: %s", request_id, user_id, str(e))
        raise HTTPException(status_code=500, detail=f"Simulation error: {str(e)}") from e

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("simulation_quib:app", host="0.0.0.0", port=8000, reload=True)
