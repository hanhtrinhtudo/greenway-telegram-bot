import os
import requests
from flask import Flask, request
from openai import OpenAI

app = Flask(__name__)

# Lấy token từ biến môi trường
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

if not TELEGRAM_TOKEN:
    raise RuntimeError("Chưa cấu hình TELEGRAM_TOKEN")
if not OPENAI_API_KEY:
    raise RuntimeError("Chưa cấu hình OPENAI_API_KEY")

TELEGRAM_API_URL = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"

# Khởi tạo OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Prompt định nghĩa vai trò chatbot
SYSTEM_PROMPT = """
Bạn là trợ lý bán hàng & chăm sóc khách hàng chuyên nghiệp của doanh nghiệp.
Yêu cầu:
- Trả lời bằng tiếng Việt, giọng điệu thân thiện, dễ hiểu.
- Hỏi lại khách khi thông tin chưa rõ.
- Hướng khách đến quyết định mua hàng, nhưng không nói quá, không hứa hẹn quá mức.
- Nếu câu hỏi ngoài phạm vi sản phẩm/dịch vụ, vẫn hỗ trợ nhưng giữ trọng tâm là giải pháp của doanh nghiệp.
"""

def send_message(chat_id: int, text: str):
    """Gửi tin nhắn về Telegram."""
    try:
        requests.post(
            f"{TELEGRAM_API_URL}/sendMessage",
            json={"chat_id": chat_id, "text": text}
        )
    except Exception as e:
        print("Lỗi gửi message về Telegram:", e)


@app.route("/", methods=["GET"])
def index():
    return "Bot is running.", 200


@app.route("/webhook", methods=["POST"])
def webhook():
    update = request.get_json(force=True, silent=True) or {}
    print("Update:", update)

    # Chỉ xử lý khi có message text
    message = update.get("message")
    if not message:
        return "no message", 200

    chat_id = message["chat"]["id"]
    text = message.get("text") or ""

    # Có thể xử lý lệnh /start, /help
    if text.startswith("/start"):
        welcome = (
            "Chào anh/chị 👋\n"
            "Em là trợ lý AI hỗ trợ tư vấn & chăm sóc khách hàng.\n"
            "Anh/chị cứ gửi câu hỏi về sản phẩm, dịch vụ, chính sách... để em hỗ trợ nhé."
        )
        send_message(chat_id, welcome)
        return "ok", 200

    # Gọi OpenAI ChatGPT
    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",  # hoặc gpt-5.1 tuỳ ngân sách :contentReference[oaicite:4]{index=4}
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": text
                }
            ],
            temperature=0.4,  # trả lời ổn định, ít "chém"
        )

        reply = completion.choices[0].message.content.strip()
    except Exception as e:
        print("Lỗi gọi OpenAI:", e)
        reply = "Hiện hệ thống AI đang bận, anh/chị vui lòng thử lại sau 1 chút nhé."

    send_message(chat_id, reply)
    return "ok", 200


if __name__ == "__main__":
    # Chạy local để test, khi deploy Render sẽ không dùng đoạn này
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))

