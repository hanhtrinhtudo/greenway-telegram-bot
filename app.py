import os
import json
from pathlib import Path

import requests
from flask import Flask, request
from openai import OpenAI

app = Flask(__name__)

# ===== LOAD DANH MỤC SẢN PHẨM =====
BASE_DIR = Path(__file__).resolve().parent
CATALOG_PATH = BASE_DIR / "data" / "welllab_catalog.json"

try:
    with open(CATALOG_PATH, "r", encoding="utf-8") as f:
        PRODUCT_CATALOG = json.load(f)
    print(f"Đã load {len(PRODUCT_CATALOG)} mục sản phẩm từ {CATALOG_PATH}")
except Exception as e:
    print("Không load được welllab_catalog.json:", e)
    PRODUCT_CATALOG = []

# ===== TOKEN & CLIENT =====
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

if not TELEGRAM_TOKEN:
    raise RuntimeError("Chưa cấu hình TELEGRAM_TOKEN")
if not OPENAI_API_KEY:
    raise RuntimeError("Chưa cấu hình OPENAI_API_KEY")

TELEGRAM_API_URL = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"
client = OpenAI(api_key=OPENAI_API_KEY)

# ===== PROMPT VAI TRÒ CHATBOT =====
SYSTEM_PROMPT = """
Bạn là trợ lý tư vấn sức khỏe & thực phẩm chức năng WELLLAB cho công ty Green Way.

Nguyên tắc:
- Trả lời bằng TIẾNG VIỆT, xưng hô lịch sự (anh/chị, em).
- Luôn dựa trên danh mục combo/sản phẩm WELLLAB được cung cấp trong ngữ cảnh.
- Giải thích cho khách hiểu đơn giản: sản phẩm giúp gì, phù hợp ai, dùng bao lâu, lưu ý gì.
- Không cam kết chữa khỏi bệnh, không thay thế đơn thuốc hoặc chẩn đoán của bác sĩ.
- Nếu khách có bệnh nền, đang mang thai, cho con bú, dùng thuốc tây → luôn khuyến cáo hỏi ý kiến bác sĩ/chuyên gia.
- Nếu câu hỏi nằm ngoài lĩnh vực sản phẩm (chuyện đời sống, tài chính…) vẫn có thể trả lời ngắn nhưng nên kéo khách quay lại chủ đề sức khỏe & sản phẩm của công ty.
"""

# ===== HÀM GỬI TIN NHẮN TELEGRAM =====
def send_message(chat_id: int, text: str):
    """Gửi tin nhắn về Telegram."""
    try:
        requests.post(
            f"{TELEGRAM_API_URL}/sendMessage",
            json={"chat_id": chat_id, "text": text}
        )
    except Exception as e:
        print("Lỗi gửi message về Telegram:", e)


# ===== TÌM COMBO / SẢN PHẨM PHÙ HỢP =====
def search_catalog(query: str, top_k: int = 5):
    """Tìm combo/sản phẩm liên quan nhất tới câu hỏi của khách (match theo keyword)."""
    if not PRODUCT_CATALOG:
        return []

    q = query.lower()
    scored = []

    for item in PRODUCT_CATALOG:
        # Các trường đem ra so sánh
        text_parts = [
            item.get("name", ""),
            " ".join(item.get("goals", [])),
            " ".join(item.get("tags", [])),
            " ".join(item.get("keywords", [])),
            item.get("who_for", "")
        ]
        haystack = " ".join(text_parts).lower()

        # Điểm = số từ khóa xuất hiện
        score = 0
        for kw in item.get("keywords", []):
            if kw.lower() in q:
                score += 3
        for g in item.get("goals", []):
            if g.lower() in q:
                score += 2

        # Thêm điểm nếu câu hỏi chứa tên combo
        name_tokens = item.get("name", "").lower().split()
        if any(t in q for t in name_tokens):
            score += 1

        if score > 0:
            scored.append((score, item))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [it for _, it in scored[:top_k]]


# ===== XÂY DỰNG CONTEXT SẢN PHẨM CHO AI =====
def build_product_context(items):
    if not items:
        return "Không tìm thấy combo cụ thể nào trong danh mục nội bộ."

    lines = ["Dưới đây là một số combo/sản phẩm trong danh mục WELLLAB liên quan tới nhu cầu của khách:"]

    for idx, it in enumerate(items, start=1):
        lines.append(f"\n[{idx}] {it.get('name','')} ({it.get('id','')})")
        goals = ", ".join(it.get("goals", []))
        if goals:
            lines.append(f"- Mục tiêu chính: {goals}")
        who_for = it.get("who_for", "")
        if who_for:
            lines.append(f"- Phù hợp cho: {who_for}")

        for p in it.get("products", []):
            lines.append(
                f"  • {p.get('label','')} – Công dụng: {p.get('benefit','')} – Cách dùng: {p.get('usage','')}"
            )

        note = it.get("notes", "")
        if note:
            lines.append(f"- Ghi chú liệu trình: {note}")

    lines.append(
        "\nKhi tư vấn, hãy CHỈ sử dụng thông tin trên, nhưng diễn đạt lại cho khách dễ hiểu, "
        "không thay thế chẩn đoán của bác sĩ và luôn khuyến cáo khách tham khảo ý kiến chuyên môn "
        "khi có bệnh lý nền."
    )
    return "\n".join(lines)


# ===== ROUTES =====
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

    # Lệnh /start
    if text.startswith("/start"):
        welcome = (
            "Chào anh/chị 👋\n"
            "Em là trợ lý AI hỗ trợ tư vấn & chăm sóc sức khỏe bằng sản phẩm WELLLAB.\n"
            "Anh/chị cứ gửi nhu cầu, triệu chứng hoặc câu hỏi về sản phẩm, liệu trình... để em hỗ trợ nhé."
        )
        send_message(chat_id, welcome)
        return "ok", 200

    # ===== TÌM SẢN PHẨM LIÊN QUAN TRONG CATALOG =====
    related_items = search_catalog(text)
    kb_context = build_product_context(related_items)

    # ===== GỌI OPENAI VỚI NGỮ CẢNH SẢN PHẨM =====
    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "system",
                    "content": (
                        "Bạn đang tư vấn dựa trên danh mục sản phẩm WELLLAB của công ty. "
                        "TUYỆT ĐỐI không bịa ra sản phẩm mới, chỉ dùng các combo/sản phẩm xuất hiện trong danh mục dưới đây.\n\n"
                        + kb_context
                    )
                },
                {
                    "role": "user",
                    "content": text
                }
            ],
            temperature=0.4,
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
