from data.milvus.indexing import MilvusIndexer
import os
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

print("\n".join(sys.path))
from llm.base import AgentClient
from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.providers.google_gla import GoogleGLAProvider

from data.cache.memory_handler import MessageMemoryHandler

import chainlit as cl

from utils.basetools import *

# Initialize Milvus indexer (run only once to create collection and index data)
# Comment this out after first run
# Replace "___________" with your collection name and FAQ file path
indexer = MilvusIndexer(collection_name="company1", faq_file="src/data/mock_data/HR_FAQ.xlsx")
indexer.run()
# Initialize model and provider
provider = GoogleGLAProvider(api_key=os.getenv("GEMINI_API_KEY"))
model = GeminiModel('gemini-2.0-flash', provider=provider)


faq_tool = create_faq_tool(collection_name="company1")
from utils.basetools.search_web_tool import SearchInput, search_web

from utils.basetools import Tool

def ptnk_search_tool(query: str, max_results: int = 5):
    search_query = f"site:ptnk.edu.vn {query}"
    input_data = SearchInput(query=search_query, max_results=max_results)
    return search_web(input_data)

def create_ptnk_tool():
    return Tool(
        name="ptnk_tool",
        func=ptnk_search_tool,
        description="Tra cứu thông tin trường PTNK từ website ptnk.edu.vn bằng search_web_tool.py",
        input_model=SearchInput,
        output_model=SearchOutput
    )

ptnk_tool = create_ptnk_tool()


# Tool tạo thời khoá biểu
from pydantic import BaseModel, Field
class TimetableInput(BaseModel):
    subjects: list = Field(..., description="Danh sách môn học và thời lượng mong muốn")
    available_times: list = Field(..., description="Các khung thời gian rảnh trong tuần")
class TimetableOutput(BaseModel):
    timetable: dict = Field(..., description="Thời khoá biểu đề xuất")
def create_timetable_tool():
    def run(input: TimetableInput) -> TimetableOutput:
        timetable = {}
        times = input.available_times.copy()
        for subject in input.subjects:
            if times:
                slot = times.pop(0)
                timetable[subject] = slot
            else:
                timetable[subject] = "Chưa có slot phù hợp"
        return TimetableOutput(timetable=timetable)
    return {
        "name": "timetable_tool",
        "description": "Tạo thời khoá biểu đề xuất dựa trên môn học và thời gian rảnh.",
        "input_model": TimetableInput,
        "output_model": TimetableOutput,
        "run": run
    }

timetable_tool = create_timetable_tool()

# Tool đọc hình ảnh (OCR)
class OCRInput(BaseModel):
    image_path: str = Field(..., description="Đường dẫn file ảnh cần đọc")
class OCROutput(BaseModel):
    text: str = Field(..., description="Nội dung trích xuất từ ảnh")
def create_ocr_tool():
    def run(input: OCRInput) -> OCROutput:
        try:
            import pytesseract
            from PIL import Image
            img = Image.open(input.image_path)
            text = pytesseract.image_to_string(img, lang='eng')
        except Exception as e:
            text = f"Lỗi đọc ảnh: {e}"
        return OCROutput(text=text)
    return {
        "name": "ocr_tool",
        "description": "Đọc và trích xuất nội dung từ file ảnh (OCR)",
        "input_model": OCRInput,
        "output_model": OCROutput,
        "run": run
    }

ocr_tool = create_ocr_tool()

# Initialize agent with tools
agent = AgentClient(
    model=model,
    system_prompt=(
        "Bạn là trợ lý AI thông minh. Hãy sử dụng `faq_tool` để tra cứu câu hỏi nội bộ, `ptnk_tool` để tra cứu thông tin từ website PTNK, `timetable_tool` để hỗ trợ tạo thời khoá biểu, và `ocr_tool` để đọc nội dung từ hình ảnh. "
        "Nếu không tìm thấy trong FAQ, hãy thử tra cứu trên website trường, hỗ trợ tạo thời khoá biểu hoặc đọc nội dung từ ảnh theo yêu cầu."
    ),
    tools=[faq_tool, ptnk_tool, timetable_tool, ocr_tool]
).create_agent()

@cl.on_chat_start
async def start():
    """Initialize chat session"""
    cl.user_session.set("message_count", 0)
    await cl.Message(content="🎓 **Chào mừng đến với Hệ thống hỗ trợ truy vấn NHÂN SỰ !**").send()
    

@cl.on_message
async def main(message: cl.Message):    
    # YOUR LOGIC HERE
    response = await agent.run((message.content))
    await cl.Message(content=str(response.output)).send()

    send_email_tool(
        EmailToolInput(
            subject="FaQ Question Received",
            body=f"Received question: {message.content}\nResponse: {response.output}"
        ), to_emails=["dung.phank24@hcmut.edu.vn"]
    )