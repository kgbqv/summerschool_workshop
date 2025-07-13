from data.milvus.indexing import MilvusIndexer
import os
from llm.base import AgentClient
from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.providers.google_gla import GoogleGLAProvider
from pydantic import BaseModel, Field
from data.cache.memory_handler import MessageMemoryHandler

import chainlit as cl
from utils.basetools import *

import requests
provider = GoogleGLAProvider(api_key=os.getenv("GEMINI_API_KEY"))
model = GeminiModel('gemini-2.0-flash', provider=provider)

def bloom_tool(user_input: str) -> str:
    """
    Calls an external API when an essay is mentioned.
    """
    api_url = "https://bloom-bert-api-dmkyqqzsta-as.a.run.app/predict"  # ✅ Replace this with your actual working URL

    headers = {
        "Content-Type": "application/json",
        # "Authorization": "Bearer YOUR_API_KEY"  # Uncomment and update if your API needs it
    }

    payload = {
        "text": user_input  # ✅ Match this key to what your API expects
    }

    try:
        response = requests.post(api_url, json=payload, headers=headers)
        print("📬 Response status code:", response.status_code)
        print("🧾 Response text:", response.text)

        response.raise_for_status()  # Raises HTTPError for bad status
        try:
            json_data = response.json()
            print("✅ Parsed JSON:", json_data)
            return f"API Response: {json_data}"
        except Exception as e:
            print("❌ Failed to parse JSON:", e)
            return f"Error: Failed to parse API response: {e}"

    except requests.exceptions.RequestException as e:
        print("❌ Network/API error:", e)
        return f"Error: Failed to reach API: {e}"

agent = AgentClient(
    model=model,  # or whichever model you prefer
    system_prompt="""
You are a helpful assistant.
When the user provide a topic and a question, use the 'bloom_tool' to find and let user know the bloom level of the question, and make a question using the topic provided with a similar bloom level
""",
    tools=[bloom_tool]
).create_agent()

@cl.on_chat_start
async def start():
    """Initialize chat session"""
    cl.user_session.set("message_count", 0)
    await cl.Message(content="welcome to the bloom rank").send()


@cl.on_message
async def main(message: cl.Message):   
   # YOUR LOGIC HERE
   response = await agent.run((message.content))
   await cl.Message(content=str(response.output)).send()