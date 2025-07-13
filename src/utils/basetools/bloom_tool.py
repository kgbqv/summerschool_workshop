from pydantic import BaseModel
from typing import Dict, Any, Optional, Union
import json
from enum import Enum
import requests

# 🟢 Trigger phrases
TRIGGERS = ["start workflow", "submit report", "send notification"]

# 🟢 Input model
class UserInput(BaseModel):
    text: str

# 🟢 Output model
class ToolResponse(BaseModel):
    triggered: bool
    message: str
    api_response_text: Optional[str] = None
    api_status_code: Optional[int] = None

# 🟢 Tool function
def agentic_post_tool(user_input: UserInput) -> ToolResponse:
    text = user_input.text.lower()

    # Detect trigger
    if any(trigger in text for trigger in TRIGGERS):
        payload = {
            "event": "triggered_action",
            "user_input": user_input.text
        }

        try:
            response = requests.post(
                url="https://bloom-bert-api-dmkyqqzsta-as.a.run.app/predict",
                json=payload,
                timeout=10
            )
            response_text = response.text

            return ToolResponse(
                triggered=True,
                message="Trigger detected. API responded.",
                api_status_code=response.status_code,
                api_response_text=response_text
            )
        except requests.RequestException as e:
            return ToolResponse(
                triggered=True,
                message=f"Trigger detected but API call failed: {str(e)}",
                api_status_code=None
            )
    else:
        return ToolResponse(
            triggered=False,
            message="No trigger detected."
        )