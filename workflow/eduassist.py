import os
import chainlit as cl
from datetime import datetime, timedelta
from utils.education_tools import *
from utils.basetools import *
from llm.base import AgentClient
from data.milvus.indexing import MilvusIndexer
from data.cache.redis_handler import RedisHandler
from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.providers.google_gla import GoogleGLAProvider

# Khởi tạo Redis
redis_handler = RedisHandler(host='localhost', port=6379, db=0)

# Khởi tạo Milvus cho các bộ dữ liệu khác nhau
def initialize_milvus_collections():
    collections = {
        "clubs": "data/clubs_data.xlsx",
        "admissions": "data/admissions_data.xlsx",
        "scores": "data/scores_data.xlsx",
        "departments": "data/departments_data.xlsx"
    }
    
    for collection_name, data_path in collections.items():
        if not MilvusIndexer.collection_exists(collection_name):
            indexer = MilvusIndexer(collection_name=collection_name, data_file=data_path)
            indexer.run()

# Khởi tạo model
provider = GoogleGLAProvider(api_key=os.getenv("GEMINI_API_KEY"))
model = GeminiModel('gemini-1.5-flash', provider=provider)

# Tạo các công cụ giáo dục
education_tools = [
    create_schedule_tool(),
    create_schedule_advisor_tool(),
    create_club_search_tool(),
    create_department_locator_tool(),
    create_progress_tracker_tool(),
    create_quiz_generator_tool(),
    create_emotion_response_tool()
]

# Tạo các công cụ cơ bản
base_tools = [
    create_faq_tool(collection_name="clubs"),
    create_faq_tool(collection_name="admissions"),
    create_faq_tool(collection_name="scores"),
    create_faq_tool(collection_name="departments"),
    create_search_web_tool(),
    create_classification_tool(),
    create_calculator_tool(),
    create_send_email_tool()
]

# Khởi tạo agent
agent = AgentClient(
    model=model,
    system_prompt="Bạn là EduBot - trợ lý giáo dục thông minh. Bạn giúp quản lý lịch học, tư vấn thông tin trường, và hỗ trợ học tập.",
    tools=education_tools + base_tools
).create_agent()

@cl.on_chat_start
async def start_chat():
    initialize_milvus_collections()
    await cl.Message(content="🎓 **Chào mừng đến với Hệ thống hỗ trợ giáo dục!**").send()
    redis_handler.set_user_state(cl.user_session.get('id'), "active", expiry=3600)

@cl.on_message
async def handle_message(message: cl.Message):
    user_id = cl.user_session.get('id')
    user_message = message.content
    
    # Lưu lịch sử hội thoại vào Redis
    redis_handler.append_chat_history(user_id, f"user: {user_message}")
    
    # Xử lý đa phương tiện
    if message.elements:
        for element in message.elements:
            if "image" in element.mime:
                # Xử lý hình ảnh (nếu cần)
                pass
    
    # Phân loại câu hỏi
    question_type = classification_tool(user_message)
    
    # Xử lý đặc biệt cho các loại câu hỏi
    if "schedule" in question_type:
        # Xử lý lịch học
        current_schedule = redis_handler.get_user_data(user_id, "schedule")
        response = await agent.run(f"{user_message}\n\nLịch hiện tại: {current_schedule}")
    elif "location" in question_type:
        # Tìm kiếm địa điểm
        response = await agent.run(user_message)
    else:
        # Xử lý chung
        response = await agent.run(user_message)
    
    # Gửi phản hồi
    await cl.Message(content=str(response.output)).send()
    
    # Lưu phản hồi vào Redis
    redis_handler.append_chat_history(user_id, f"assistant: {response.output}")

@cl.on_schedule
async def check_schedule_reminders():
    # Kiểm tra và gửi nhắc nhở lịch học
    all_users = redis_handler.get_all_users()
    now = datetime.now()
    
    for user_id in all_users:
        schedule = redis_handler.get_user_data(user_id, "schedule")
        if schedule:
            for event in schedule:
                event_time = datetime.strptime(event['time'], "%Y-%m-%d %H:%M")
                if now < event_time < now + timedelta(minutes=10):
                    await cl.Message(
                        content=f"⏰ Nhắc nhở: Sắp đến giờ {event['subject']} vào lúc {event['time']}",
                        to=user_id
                    ).send()

@cl.action_callback("lock_app")
async def on_action(action: cl.Action):
    # Xử lý khóa ứng dụng
    lock_time = action.value
    redis_handler.set_user_data(
        cl.user_session.get('id'), 
        "lock_settings",
        {"lock_time": lock_time, "status": "active"}
    )
    await action.remove()
    await cl.Message(content=f"🔒 Ứng dụng sẽ bị khóa vào lúc {lock_time}").send()