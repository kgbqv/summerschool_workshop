from data.milvus.indexing import MilvusIndexer
import os
from llm.base import AgentClient
from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.providers.google_gla import GoogleGLAProvider

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
   model=model,
   system_prompt="You are a friendly virtual assistant. If the user specifies answering a question, do it. Otherwise, use the 'bloom_tool' to get the bloom level of each question in the file and make a question with the same bloom level in the topic provided by the user.",  # Replace with your system prompt
   tools = [bloom_tool]
).create_agent()



@cl.on_message
async def main(message: cl.Message):
    # Now you have the file path:
    #await cl.Message(content=f"✅ File saved to: {read_file_tool(file.path)}").send()
    if message.elements:
        file_elements = [e for e in message.elements if isinstance(e, cl.File)]
        file = file_elements[0]
        filecontent=read_file_tool(file.path)
        response = await agent.run((message.content+"<file start>"+filecontent.content+"<file end>"))
        await cl.Message(content=str(response.output)).send()
    else:
        response = await agent.run((message.content))
        await cl.Message(content=str(response.output)).send()
