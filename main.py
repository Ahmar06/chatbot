from agents import Agent, OpenAIChatCompletionsModel,AsyncOpenAI,Runner,RunConfig
from dotenv import load_dotenv
import os
import asyncio
from openai.types.responses import ResponseTextDeltaEvent
import chainlit as cl



load_dotenv()

MODEL_NAME = "gemini-2.0-flash"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

external_client = AsyncOpenAI(
    api_key = GEMINI_API_KEY,
    base_url = "https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    model = MODEL_NAME,
    openai_client = external_client,    
)

config = RunConfig(
    model = model,
    model_provider = external_client,
    tracing_disabled = True,
)

agent = Agent(
    name = "Assistant",
    instructions ="You are a coading assistant",
    model = model,
)

@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("history",[])
    await cl.Message(content="Hello, I am your coding assistant. How can I help you today?").send() 

@cl.on_message
async def handle_message(message:cl.Message):
    history = cl.user_session.get("history", [])
    msg = cl.Message(content="")
    await msg.send()
    history.append({"role":"user","content":message.content})
    result = Runner.run_streamed(
        agent,
        input = history,
        run_config = config
    )
    async for event in result.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            await msg.stream_token(event.data.delta)
    history.append({"role":"assistant","content":result.final_output})
    cl.user_session.set("history", history)
