from agents import Agent,OpenAIChatCompletionsModel,Runner,RunConfig,AsyncOpenAI
import os 
from dotenv import load_dotenv
import asyncio

load_dotenv()

while True:
    async def main():

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
          tracing_disabled = True,
          model = model,
          model_provider = external_client,
        ) 

        agent = Agent(
            name = "Assistant",
            instructions = "You are a helpful assistant Answer academic questions Provide study tips Summarize small text passages",
            model = model,
        )
        result = await Runner.run(
            agent,
            input = input("Enter your query: "),
            run_config = config,
        )

        print(result.final_output)

    asyncio.run(main())