import os
import chainlit as cl
from llm.base import AgentClient
from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.providers.google_gla import GoogleGLAProvider

from data.milvus.indexing import MilvusIndexer
from data.cache.redis_handler import RedisCacheHandler  # New Redis handler

# Import all tools including the new Bloom tool
from utils.basetools import *

# Initialize Redis cache
redis_cache = RedisCacheHandler(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    db=int(os.getenv("REDIS_DB", 0)),
    ttl=3600  # 1 hour cache
)

# Initialize Milvus (run once)
# indexer = MilvusIndexer(collection_name="ptnk_clubs", file_path="data/pho_thong_nang_khieu.pdf")
# indexer.run()

# Initialize models
provider = GoogleGLAProvider(api_key=os.getenv("GEMINI_API_KEY"))
model = GeminiModel('gemini-1.5-flash', provider=provider)

# Initialize tools
faq_tool = create_faq_tool(collection_name="ptnk_clubs")
search_tool = create_search_in_file_tool(file_path="data/pho_thong_nang_khieu.pdf")
email_tool = create_send_email_tool(default_from="assistant@ptnk.edu.vn")

tools = [
    faq_tool,
    search_tool,
    email_tool,
    Tool(
        name="BloomDifficulty",
        func=bloom_difficulty_tool,
        args_schema=BloomToolInput,
        description="Assess cognitive difficulty level using Bloom's taxonomy"
    ),
    Tool(
        name="AdmissionCalculator",
        func=calculate_admission_score,
        args_schema=AdmissionInput,
        description="Calculate admission scores based on subject scores"
    )
]

# Create agent
agent = AgentClient(
    model=model,
    system_prompt="""
    Bạn là trợ lý ảo cho Trường Phổ Thông Năng Khiếu (ĐHQG-HCM). 
    Hãy trả lời câu hỏi về: 
    - Danh sách câu lạc bộ
    - Thông tin tuyển sinh
    - Điểm chuẩn các năm
    Sử dụng công cụ thích hợp cho từng loại câu hỏi.
    Đánh giá độ khó câu hỏi bằng BloomDifficulty khi cần.
    """,
    tools=tools
).create_agent()

@cl.on_chat_start
async def start_chat():
    settings = {
        "model": "gemini-1.5-flash",
        "temperature": 0.3,
        "max_tokens": 2048
    }
    cl.user_session.set("settings", settings)
    await cl.Message(content="🎓 Chào mừng đến với Hệ thống hỗ trợ Trường PTNK!").send()

@cl.on_message
async def handle_message(message: cl.Message):
    # Check cache first
    cached_response = redis_cache.get(message.content)
    if cached_response:
        await cl.Message(content=cached_response).send()
        return
    
    # Assess difficulty
    bloom_input = BloomToolInput(text=message.content)
    bloom_result = bloom_difficulty_tool(bloom_input)
    
    # Add difficulty context to query
    enhanced_query = f"[Độ khó: {bloom_result.difficulty.value}] {message.content}"
    
    # Process query
    response = await agent.run(enhanced_query)
    
    # Cache and send response
    redis_cache.set(message.content, str(response.output))
    await cl.Message(content=str(response.output)).send()
    
    # Send notification for complex queries
    if bloom_result.difficulty in [BloomLevel.ANALYZE, BloomLevel.EVALUATE, BloomLevel.CREATE]:
        email_tool.run(EmailToolInput(
            subject="Câu hỏi phức tạp cần xem xét",
            body=f"Người dùng: {message.author}\nCâu hỏi: {message.content}\nĐộ khó: {bloom_result.difficulty.value}"
        ), to_emails=["admin@ptnk.edu.vn"])

# New admission calculator tool
class AdmissionInput(BaseModel):
    math: float
    literature: float
    english: float
    specialized: float

def calculate_admission_score(input: AdmissionInput) -> str:
    """Calculate admission score: (math + lit + eng) + specialized*2"""
    base_score = input.math + input.literature + input.english
    specialized_score = input.specialized * 2
    total = base_score + specialized_score
    return f"Điểm xét tuyển: {total:.2f}\n(Toán: {input.math}, Văn: {input.literature}, Anh: {input.english}, Chuyên: {input.specialized}*2)"