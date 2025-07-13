import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

print("\n".join(sys.path))
from data.milvus.indexing import MilvusIndexer
import os
import json
import redis
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from llm.base import AgentClient
from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.providers.google_gla import GoogleGLAProvider
import chainlit as cl
from utils.basetools import *
from pydantic import BaseModel, Field

# Initialize Redis
redis_client = redis.Redis(host='localhost', port=6379, db=0)

# Initialize Milvus indexers (run once)
# indexer1 = MilvusIndexer(collection_name="clubs", faq_file="data/clubs_data.xlsx")
# indexer1.run()
# indexer2 = MilvusIndexer(collection_name="admissions", faq_file="data/admissions_data.xlsx")
# indexer2.run()

# Initialize model
provider = GoogleGLAProvider(api_key=os.getenv("GEMINI_API_KEY"))
model = GeminiModel('gemini-2.0-pro', provider=provider)

# --- New Tools Definitions ---
class ScheduleInput(BaseModel):
    subjects: List[str] = Field(..., description="Danh sách môn học")
    free_slots: Dict[str, List[str]] = Field(..., description="Khung giờ rảnh dạng {'Thứ': ['9:00-11:00']}")

class ScheduleAdviceInput(BaseModel):
    current_schedule: Dict[str, List[str]] = Field(..., description="Thời khóa biểu hiện tại")

class LocationSearchInput(BaseModel):
    query: str = Field(..., description="Nhu cầu tìm kiếm địa điểm")

class QuizInput(BaseModel):
    material: str = Field(..., description="Nội dung tài liệu ôn tập")

class ProgressInput(BaseModel):
    completed: List[str] = Field(..., description="Các bài đã hoàn thành")
    total: List[str] = Field(..., description="Toàn bộ bài tập cần làm")

def schedule_planner_tool(input: ScheduleInput) -> str:
    """
    Tạo thời khóa biểu mới dựa trên môn học và khung giờ rảnh
    """
    # Logic tạo lịch học (có thể kết hợp LLM)
    schedule = {}
    for day, slots in input.free_slots.items():
        schedule[day] = []
        for i, slot in enumerate(slots):
            if i < len(input.subjects):
                subject = input.subjects[i]
                schedule[day].append(f"{slot} - Môn: {subject}")
    
    # Lưu vào Redis
    user_id = cl.user_session.get("id")
    redis_client.set(f"schedule:{user_id}", json.dumps(schedule))
    
    return json.dumps(schedule, ensure_ascii=False)

def schedule_advisor_tool(input: ScheduleAdviceInput) -> str:
    """
    Phân tích và đề xuất cải thiện thời khóa biểu
    """
    # Logic phân tích lịch học
    advice = "Đề xuất:\n"
    for day, tasks in input.current_schedule.items():
        if len(tasks) > 3:
            advice += f"- {day}: Giảm tải từ {len(tasks)} xuống 3 môn\n"
    
    # Kiểm tra nghỉ ngơi
    advice += "\nLưu ý: Nên có ít nhất 30 phút nghỉ giữa các môn học"
    return advice

def location_search_tool(input: LocationSearchInput) -> str:
    """
    Tìm kiếm địa điểm trong trường
    """
    # Sử dụng Milvus để tìm kiếm
    results = search_relevant_document_tool(
        SearchRelevantDocumentToolInput(
            query=input.query, 
            collection_name="locations"
        )
    )
    return results[:3]  # Trả về top 3 kết quả

def quiz_generator_tool(input: QuizInput) -> str:
    """
    Tạo câu hỏi ôn tập từ tài liệu
    """
    # Sử dụng LLM để tạo câu hỏi
    prompt = f"Tạo 5 câu hỏi trắc nghiệm từ tài liệu sau:\n{input.material}"
    quiz = model.generate(prompt)
    return quiz

def progress_tracker_tool(input: ProgressInput) -> str:
    """
    Theo dõi tiến độ học tập
    """
    progress = len(input.completed) / len(input.total) * 100
    return f"Tiến độ: {progress:.2f}% - Đã hoàn thành {len(input.completed)}/{len(input.total)} bài"

def emotion_response_tool(emotion: str) -> str:
    """
    Phản hồi theo cảm xúc
    """
    responses = {
        "vui": ["Tuyệt vời! 😊", "Thật tốt khi bạn đang vui!", "Hãy duy trì tinh thần này nhé!"],
        "buồn": ["Mình ở đây để giúp bạn 🤗", "Mọi chuyện rồi sẽ tốt thôi...", "Bạn muốn chia sẻ thêm không?"],
        "căng thẳng": ["Hít thở sâu nào... 🌬️", "Bạn có muốn nghe nhạc thư giãn?", "Hãy nghỉ ngơi một chút"]
    }
    return random.choice(responses.get(emotion.lower(), ["Mình luôn sẵn sàng lắng nghe bạn ❤️"]))

# Initialize all tools
tools = [
    create_faq_tool(collection_name="clubs"),
    create_faq_tool(collection_name="admissions"),
    create_faq_tool(collection_name="benchmarks"),
    Tool(name="schedule_planner", func=schedule_planner_tool, args_schema=ScheduleInput),
    Tool(name="schedule_advisor", func=schedule_advisor_tool, args_schema=ScheduleAdviceInput),
    Tool(name="location_search", func=location_search_tool, args_schema=LocationSearchInput),
    Tool(name="quiz_generator", func=quiz_generator_tool, args_schema=QuizInput),
    Tool(name="progress_tracker", func=progress_tracker_tool, args_schema=ProgressInput),
    Tool(name="emotion_response", func=emotion_response_tool)
]

# Initialize agent
agent = AgentClient(
    model=model,
    system_prompt="""
    Bạn là trợ lý giáo dục thông minh với các chức năng:
    1. Quản lý lịch học, tạo thời khóa biểu
    2. Nhắc nhở học tập 10p trước giờ học
    3. Tư vấn cải thiện thời khóa biểu
    4. Trả lời thông tin trường học, CLB
    5. Tra cứu địa điểm trong trường
    6. Tạo câu hỏi ôn tập
    7. Theo dõi tiến độ học tập
    8. Tương tác đa phương tiện và cảm xúc
    
    Luôn sử dụng công cụ phù hợp cho từng tác vụ.
    """,
    tools=tools
).create_agent()

# Background task for reminders
async def check_reminders():
    while True:
        user_id = cl.user_session.get("id")
        if schedule := redis_client.get(f"schedule:{user_id}"):
            schedule = json.loads(schedule)
            now = datetime.now()
            current_day = now.strftime("%A")
            current_time = now.strftime("%H:%M")
            
            for task in schedule.get(current_day, []):
                start_time = task.split("-")[0].strip()
                task_time = datetime.strptime(start_time, "%H:%M")
                reminder_time = task_time - timedelta(minutes=10)
                
                if now >= reminder_time and now < task_time:
                    await cl.Message(
                        content=f"⏰ Nhắc nhở: Bạn có môn học bắt đầu lúc {start_time}!"
                    ).send()
        await asyncio.sleep(60)  # Kiểm tra mỗi phút

@cl.on_chat_start
async def start():
    cl.user_session.set("id", str(uuid.uuid4()))
    asyncio.create_task(check_reminders())
    await cl.Message(content="🎓 Chào mừng đến với Hệ thống hỗ trợ học tập!").send()

@cl.on_message
async def main(message: cl.Message):
    # Xử lý cảm xúc
    if any(keyword in message.content.lower() for keyword in ["vui", "buồn", "căng thẳng"]):
        emotion = [k for k in ["vui", "buồn", "căng thẳng"] if k in message.content.lower()][0]
        response = emotion_response_tool(emotion)
        await cl.Message(content=response).send()
        return
    
    # Xử lý yêu cầu thông thường
    response = await agent.run(message.content)
    await cl.Message(content=str(response.output)).send()