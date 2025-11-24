import os
import json
import re
import unicodedata
from pathlib import Path
from datetime import datetime

import requests
from flask import Flask, request
from openai import OpenAI

app = Flask(__name__)

# ========= ĐƯỜNG DẪN & DATA =========
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"


def load_json(path, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Không load được {path}: {e}")
        return default


CATALOG_PATH = DATA_DIR / "welllab_catalog.json"       # danh mục combo
SYMPTOMS_PATH = DATA_DIR / "symptoms_mapping.json"     # intent -> combo
FAQ_PATH = DATA_DIR / "faq.json"                       # câu hỏi thường gặp
OBJECTIONS_PATH = DATA_DIR / "objections.json"         # từ chối phổ biến
USERS_PATH = DATA_DIR / "users_store.json"             # hồ sơ người dùng
PRODUCTS_PATH = DATA_DIR / "welllab_products.json"     # danh mục sản phẩm lẻ

WELLLAB_CATALOG = load_json(CATALOG_PATH, [])
SYMPTOM_RULES = load_json(SYMPTOMS_PATH, [])
FAQ_LIST = load_json(FAQ_PATH, [])
OBJECTION_LIST = load_json(OBJECTIONS_PATH, [])
WELLLAB_PRODUCTS = load_json(PRODUCTS_PATH, [])


# ========= TIỆN ÍCH CHUẨN HÓA =========
def normalize_text(s: str) -> str:
    """Bỏ dấu, về thường để so khớp linh hoạt hơn."""
    if not s:
        return ""
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return s.lower().strip()


def search_combo_by_text(query: str, top_k: int = 1) -> list[dict]:
    """
    Tìm combo theo tên / alias trong welllab_catalog.json.
    So khớp không dấu, không phân biệt hoa thường.
    """
    q = normalize_text(query)
    if not q or not WELLLAB_CATALOG:
        return []

    results: list[tuple[int, dict]] = []
    for combo in WELLLAB_CATALOG:
        name = normalize_text(combo.get("name", ""))
        aliases = [normalize_text(a) for a in combo.get("aliases", [])]
        haystack = " ".join([name] + aliases)

        score = 0
        for token in q.split():
            if token and token in haystack:
                score += 1

        if score > 0:
            results.append((score, combo))

    results.sort(key=lambda x: x[0], reverse=True)
    return [c for score, c in results[:top_k]]


def search_product_by_text(query: str, top_k: int = 1) -> list[dict]:
    """
    Tìm sản phẩm theo tên / mã trong welllab_products.json.
    """
    q = normalize_text(query)
    if not q or not WELLLAB_PRODUCTS:
        return []

    results: list[tuple[int, dict]] = []
    for prod in WELLLAB_PRODUCTS:
        name = normalize_text(prod.get("name", ""))
        code = normalize_text(prod.get("code", ""))
        aliases = [name, code]
        haystack = " ".join([a for a in aliases if a])

        score = 0
        for token in q.split():
            if token and token in haystack:
                score += 1
        if score > 0:
            results.append((score, prod))

    results.sort(key=lambda x: x[0], reverse=True)
    return [p for score, p in results[:top_k]]


# ========= USER STORE =========
def load_users_store():
    try:
        with open(USERS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_users_store(store: dict):
    try:
        with open(USERS_PATH, "w", encoding="utf-8") as f:
            json.dump(store, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print("Lỗi lưu users_store.json:", e)


USERS_STORE = load_users_store()

# ========= LOG HỘI THOẠI =========
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)
CONV_LOG_PATH = LOG_DIR / "conversations.log"


def get_now_iso():
    try:
        return datetime.now().isoformat(timespec="seconds")
    except Exception:
        return datetime.now().isoformat()


def log_event(user_id: int, direction: str, text: str, extra: dict | None = None):
    rec: dict = {
        "ts": get_now_iso(),
        "user_id": user_id,
        "direction": direction,
        "text": text,
    }
    if extra:
        rec["meta"] = extra
    try:
        with open(CONV_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception as e:
        print("Lỗi ghi log hội thoại:", e)


# ========= HỒ SƠ NGƯỜI DÙNG =========
def get_or_create_user_profile(telegram_user_id: int, tg_user: dict) -> dict:
    uid = str(telegram_user_id)
    profile = USERS_STORE.get(uid) or {
        "telegram_id": telegram_user_id,
        "first_seen": get_now_iso(),
        "last_seen": get_now_iso(),
        "name": "",
        "username": "",
        "main_needs": {},
        "intents_count": {},
        "total_messages": 0,
        "notes": "",
    }

    if tg_user:
        uname = (tg_user.get("username") or "").strip()
        fname = (tg_user.get("first_name") or "").strip()
        lname = (tg_user.get("last_name") or "").strip()
        full_name = (fname + " " + lname).strip()
        if full_name:
            profile["name"] = full_name
        if uname:
            profile["username"] = uname

    profile["last_seen"] = get_now_iso()
    USERS_STORE[uid] = profile
    return profile


def touch_user_stats(profile: dict, need: str | None = None, intent: str | None = None):
    profile["total_messages"] = int(profile.get("total_messages") or 0) + 1

    if need:
        needs = profile.get("main_needs") or {}
        needs[need] = int(needs.get(need) or 0) + 1
        profile["main_needs"] = needs

    if intent:
        intents = profile.get("intents_count") or {}
        intents[intent] = int(intents.get(intent) or 0) + 1
        profile["intents_count"] = intents

    save_users_store(USERS_STORE)


# ========= TELEGRAM & OPENAI =========
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

if not TELEGRAM_TOKEN:
    raise RuntimeError("Chưa cấu hình TELEGRAM_TOKEN")
if not OPENAI_API_KEY:
    raise RuntimeError("Chưa cấu hình OPENAI_API_KEY")

TELEGRAM_API_URL = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"
client = OpenAI(api_key=OPENAI_API_KEY)

# ========= SESSION THEO CHAT =========
SESSIONS: dict[int, dict] = {}


def get_session(chat_id: int) -> dict:
    s = SESSIONS.get(chat_id)
    if not s:
        s = {
            "mode": "tvv",          # default: hỗ trợ TƯ VẤN VIÊN
            "intent": None,
            "profile": {},
            "stage": "await_need",
            "first_issue": None,
            "need": None,
            "last_combo": None,
            "last_product": None,
            "clarify_rounds": 0,
        }
        SESSIONS[chat_id] = s
    return s


# ========= PROMPT HỆ THỐNG =========
BASE_SYSTEM_PROMPT = (
    "Bạn là TRỢ LÝ AI NỘI BỘ cho đội ngũ TƯ VẤN VIÊN của công ty Con Đường Xanh (WELLLAB).\n"
    "Người đang nhắn với bạn là TƯ VẤN VIÊN, không phải khách hàng cuối.\n\n"
    "NHIỆM VỤ CHÍNH:\n"
    "- Giúp tư vấn viên hiểu rõ từng combo/sản phẩm, đối tượng dùng, cách giải thích đơn giản cho khách.\n"
    "- Hướng dẫn tư vấn viên đặt câu hỏi khai thác nhu cầu, gợi ý kịch bản tư vấn và kịch bản chốt đơn.\n"
    "- Gợi ý cách xử lý từ chối/lo lắng của khách một cách tinh tế, tôn trọng, tuân thủ quy định.\n"
    "- Chỉ sử dụng các combo/sản phẩm có trong ngữ cảnh nội bộ, không bịa thêm.\n\n"
    "CÁCH TRẢ LỜI:\n"
    "- Trả lời ngắn gọn, rõ ý, ưu tiên bullet.\n"
    "- Thường chia thành 3–4 phần: (1) Tóm tắt case khách; "
    "(2) Gợi ý câu hỏi tư vấn viên nên hỏi; "
    "(3) Gợi ý combo/sản phẩm phù hợp và cách GIẢI THÍCH CHO KHÁCH; "
    "(4) Gợi ý 1–2 câu chốt mềm.\n"
    "- Xưng hô với người đang chat là 'anh/chị' (tư vấn viên). Không nói như đang chat trực tiếp với khách.\n"
)


# ========= LỜI CHÀO / MENU =========
def build_welcome_message() -> str:
    return (
        "Chào anh/chị 👋\n"
        "Em là trợ lý AI nội bộ hỗ trợ *TƯ VẤN VIÊN* tư vấn & chăm sóc sức khỏe bằng sản phẩm WELLLAB.\n\n"
        "Anh/chị có thể dùng em để:\n"
        "- Phân tích case khách (triệu chứng, bệnh nền, nhu cầu...)\n"
        "- Hỏi về combo/sản phẩm cụ thể\n"
        "- Hỏi cách xử lý từ chối, chính sách, kịch bản chốt đơn\n\n"
        "Anh/chị cứ mô tả case khách hoặc gõ tên combo/sản phẩm, em sẽ hỗ trợ hết sức. 💚"
    )


def get_main_menu_keyboard():
    return [
        ["🧠 Phân tích case khách"],
        ["🧴 Hỏi combo / sản phẩm"],
        ["🛡 Chính sách & xử lý từ chối"],
    ]


# ========= INTENT & NEED =========
INTENT_PRIORITY_DEFAULT = 10


def get_intent_priority(intent: str) -> int:
    for rule in SYMPTOM_RULES:
        if rule.get("intent") == intent:
            return int(rule.get("priority", INTENT_PRIORITY_DEFAULT))
    return INTENT_PRIORITY_DEFAULT


def detect_intent_from_text(text: str) -> str | None:
    t = text.lower()
    best_intent = None
    best_score = 0

    for rule in SYMPTOM_RULES:
        intent = rule.get("intent")
        kws = rule.get("keywords", [])
        matches = 0
        for kw in kws:
            kw_l = kw.lower().strip()
            if kw_l and kw_l in t:
                matches += 1

        if matches > 0:
            priority = get_intent_priority(intent)
            score = matches * 10 + priority
            if score > best_score:
                best_score = score
                best_intent = intent

    return best_intent


def detect_need(text: str) -> str:
    t = text.lower()

    health_kws = [
        "đau ", "bị đau", "benh", "bệnh", "triệu chứng", "huyết áp", "tiểu đường",
        "mỡ máu", "gan", "thận", "da cơ địa", "vảy nến", "mất ngủ", "khó ngủ", "ho", "khó thở",
        "viêm", "ngứa", "mụn", "sức khỏe", "suc khoe",
    ]
    product_kws = [
        "sản phẩm", "san pham", "combo", "liệu trình", "lieu trinh", "loại nào", "dùng gì",
        "công dụng", "thành phần", "uống như thế nào", "cách dùng", "bao lâu",
        "giá bao nhiêu", "bao nhiêu tiền",
    ]
    policy_kws = [
        "mua hàng", "dat hang", "đặt hàng", "mua ở đâu", "ship", "giao hàng",
        "thanh toán", "thanh toan", "chuyển khoản", "cod", "đổi trả", "bảo hành",
        "bao hanh", "chính sách",
    ]

    if any(kw in t for kw in health_kws):
        return "health"
    if any(kw in t for kw in product_kws):
        return "product"
    if any(kw in t for kw in policy_kws):
        return "policy"
    return "other"


# ========= KHÔNG CÓ VẤN ĐỀ SỨC KHOẺ =========
NO_HEALTH_PATTERNS = [
    "không", "ko", "k", "khong", "hong", "hông",
    "không có", "ko có", "k có",
    "không bị", "ko bị", "k bị",
    "không vấn đề", "k vấn đề", "ko vấn đề",
    "không sao", "ko sao", "k sao",
]


def is_no_health_intent(text: str) -> bool:
    t = text.lower().strip()
    if t in ["không", "ko", "k", "khong"]:
        return True
    for p in NO_HEALTH_PATTERNS:
        if t == p or t.startswith(p + " "):
            return True
    return False


# ========= CHỌN COMBO TỪ INTENT =========
def choose_combo(intent: str | None) -> dict | None:
    if not intent:
        return None
    rule = next((r for r in SYMPTOM_RULES if r.get("intent") == intent), None)
    if not rule:
        return None
    preferred_names = rule.get("preferred_combos", [])
    for name in preferred_names:
        combo = next((c for c in WELLLAB_CATALOG if c.get("name") == name), None)
        if combo:
            return combo
    return None


# ========= TRÍCH HỒ SƠ TỪ VĂN BẢN =========
def extract_profile(text: str) -> dict:
    profile: dict = {}
    lower = text.lower()

    m_age = re.search(r"(\d{2})\s*t[uô]i", lower)
    if m_age:
        try:
            profile["age"] = int(m_age.group(1))
        except Exception:
            pass

    if "nam" in lower:
        profile["gender"] = "nam"
    if "nữ" in lower or "nu" in lower:
        profile["gender"] = "nữ"

    if "không bệnh nền" in lower or "ko bệnh nền" in lower or "k bệnh nền" in lower:
        profile["has_chronic"] = False
    elif "bệnh nền" in lower:
        profile["has_chronic"] = True

    return profile


# ========= FAQ & OBJECTIONS =========
def match_keywords_any(text: str, keywords: list[str]) -> bool:
    t = text.lower()
    for kw in keywords:
        if kw.lower() in t:
            return True
    return False


def try_answer_faq(text: str) -> str | None:
    for item in FAQ_LIST:
        kws = item.get("keywords_any", [])
        if kws and match_keywords_any(text, kws):
            return item.get("answer")
    return None


def try_answer_objection(text: str) -> str | None:
    for item in OBJECTION_LIST:
        kws = item.get("keywords_any", [])
        if kws and match_keywords_any(text, kws):
            return item.get("answer")
    return None


# ========= CONTEXT GỬI OPENAI =========
def build_combo_context(combo: dict | None) -> str:
    if not combo:
        return "Hiện chưa xác định được combo cụ thể."

    lines: list[str] = []
    lines.append(f"Combo: {combo.get('name', '')}")
    header = combo.get("header_text", "")
    if header:
        lines.append("\n[Thông tin]:")
        lines.append(header)

    duration = combo.get("duration_text", "")
    if duration:
        lines.append("\n[Thời gian liệu trình khuyến nghị]:")
        lines.append(duration)

    prods = combo.get("products", [])
    if prods:
        lines.append("\n[Thành phần combo]:")
        for idx, p in enumerate(prods, start=1):
            name = p.get("name", "")
            text = p.get("text", "")
            code = p.get("code", "")
            url_p = p.get("url") or p.get("link") or ""
            line = f"{idx}. {name}"
            if code:
                line += f" ({code})"
            if text:
                line += f": {text}"
            if url_p:
                line += f" [LINK: {url_p}]"
            lines.append(line)
    return "\n".join(lines)


def build_product_context(prod: dict | None) -> str:
    if not prod:
        return "Chưa có sản phẩm cụ thể."
    name = prod.get("name", "")
    code = prod.get("code", "")
    price = prod.get("price", "")
    ingredients = prod.get("ingredients", "")
    usage = prod.get("usage", "")
    benefits = prod.get("benefits", "")
    link = prod.get("link", "")
    lines = [
        f"Tên: {name}",
        f"Mã: {code}",
        f"Giá: {price}",
        f"Thành phần: {ingredients}",
        f"Cách dùng: {usage}",
        f"Lợi ích chính: {benefits}",
        f"Link: {link}",
    ]
    return "\n".join(lines)


def build_profile_context(profile: dict) -> str:
    if not profile:
        return "Chưa có thêm thông tin cụ thể về tuổi, giới tính hay bệnh nền."
    parts: list[str] = []
    if profile.get("age"):
        parts.append(f"Tuổi khoảng: {profile['age']}.")
    if profile.get("gender"):
        parts.append(f"Giới tính: {profile['gender']}.")
    if profile.get("has_chronic") is True:
        parts.append("Có bệnh nền (chi tiết chưa rõ).")
    elif profile.get("has_chronic") is False:
        parts.append("Không có bệnh nền.")
    return " ".join(parts)


def format_combo_for_tvv(combo: dict) -> str:
    """Đoạn text cố định cho TVV: combo + link từng sản phẩm."""
    if not combo:
        return "Hiện chưa xác định được combo cụ thể ạ."

    name = combo.get("name", "")
    header = combo.get("header_text", "")
    duration = combo.get("duration_text", "")
    prods = combo.get("products", [])

    lines: list[str] = []
    lines.append(f"*Combo đề xuất:* *{name}*")
    if header:
        lines.append(f"> {header}")
    if duration:
        lines.append(f"- Thời gian liệu trình khuyến nghị: {duration}")

    if prods:
        lines.append("\n*Các sản phẩm trong combo (kèm link để gửi khách):*")
        for idx, p in enumerate(prods, start=1):
            pname = p.get("name", "")
            code = p.get("code", "")
            url_p = p.get("url") or p.get("link") or ""
            note = p.get("short_text") or p.get("text", "")
            line = f"{idx}. *{pname}*"
            if code:
                line += f" ({code})"
            if note:
                line += f": {note}"
            if url_p:
                line += f"\n   Link: {url_p}"
            lines.append(line)

    combo_url = combo.get("combo_url") or combo.get("url") or combo.get("link") or ""
    if combo_url:
        lines.append(f"\n*Link combo tổng:* {combo_url}")

    return "\n".join(lines)


def format_product_for_tvv(prod: dict) -> str:
    """Đoạn text cố định: thông tin chi tiết sản phẩm + link."""
    if not prod:
        return "Hiện chưa xác định được sản phẩm cụ thể ạ."

    name = prod.get("name", "")
    code = prod.get("code", "")
    price = prod.get("price", "")
    ingredients = prod.get("ingredients", "")
    usage = prod.get("usage", "")
    benefits = prod.get("benefits", "")
    link = prod.get("link", "")

    lines = [
        f"*Sản phẩm:* *{name}* ({code})",
    ]
    if price:
        lines.append(f"- Giá tham khảo: {price}")
    if benefits:
        lines.append(f"- Công dụng chính: {benefits}")
    if ingredients:
        lines.append(f"- Thành phần chính: {ingredients}")
    if usage:
        lines.append(f"- Cách dùng khuyến nghị: {usage}")
    if link:
        lines.append(f"- Link sản phẩm (gửi khách): {link}")

    return "\n".join(lines)


# ========= CÂU HỎI LÀM RÕ =========
CLARIFY_QUESTIONS = {
    "blood_pressure": (
        "Để tư vấn chính xác hơn về *huyết áp* cho KH, anh/chị nên khai thác thêm:\n"
        "- KH bị cao huyết áp lâu chưa, đã được bác sĩ chẩn đoán hay tự đo ở nhà?\n"
        "- Hiện tại KH có đang dùng thuốc huyết áp đều đặn không?\n"
        "- KH có kèm đau đầu, chóng mặt, khó thở hay đau ngực không?"
    ),
    "diabetes": (
        "Với *tiểu đường*, anh/chị có thể hỏi thêm KH:\n"
        "- Được chẩn đoán type mấy và bao lâu rồi?\n"
        "- Đường huyết gần đây đo được khoảng bao nhiêu?\n"
        "- KH có đang dùng thuốc hay tiêm insulin không?"
    ),
    "weight_loss": (
        "Với case *thừa cân, béo phì*, nên hỏi:\n"
        "- Chiều cao, cân nặng hiện tại?\n"
        "- Tăng cân lâu chưa, đã từng giảm nhưng bị tăng lại không?\n"
        "- Chế độ ăn uống và vận động hiện tại của KH như thế nào?"
    ),
    "digestive": (
        "Về *tiêu hoá*, anh/chị khai thác thêm:\n"
        "- KH hay bị đầy bụng, ợ hơi, ợ chua hay táo bón/tiêu chảy?\n"
        "- Triệu chứng kéo dài bao lâu, đã từng nội soi hoặc khám dạ dày chưa?\n"
        "- Thói quen ăn uống có thất thường, dùng nhiều rượu bia/cà phê không?"
    ),
    "respiratory": (
        "Với *hô hấp*, nên hỏi:\n"
        "- KH ho khan, ho có đờm hay khó thở, khò khè?\n"
        "- Triệu chứng kéo dài bao lâu, có tái lại theo mùa không?\n"
        "- KH có hút thuốc hoặc làm việc trong môi trường khói bụi không?"
    ),
    "skin_psoriasis": (
        "Với *viêm da cơ địa/vảy nến*, nên hỏi:\n"
        "- Tình trạng da: đỏ rát, bong vảy, ngứa nhiều hay chỉ khô nứt?\n"
        "- Vị trí tổn thương: tay, chân, thân mình hay lan rộng?\n"
        "- KH đã từng dùng thuốc bôi/uống da liễu nào, có bệnh dị ứng kèm theo không?"
    ),
    "default": (
        "Để em hỗ trợ chọn combo/phác đồ tốt hơn, anh/chị có thể khai thác KH thêm:\n"
        "- Triệu chứng chính là gì và kéo dài bao lâu?\n"
        "- Tuổi, giới tính, bệnh nền, thuốc đang dùng?\n"
        "- Mục tiêu là giảm triệu chứng nhanh, phòng tái phát hay nâng sức khoẻ tổng thể?"
    ),
}


def get_clarify_question(intent: str | None) -> str:
    if not intent:
        return CLARIFY_QUESTIONS["default"]
    return CLARIFY_QUESTIONS.get(intent, CLARIFY_QUESTIONS["default"])


# ========= GỌI OPENAI =========
def call_openai_for_answer(
    user_text: str,
    session: dict,
    combo: dict | None = None,
    product: dict | None = None,
) -> str:
    mode = session.get("mode", "tvv")
    intent = session.get("intent")
    profile = session.get("profile", {})

    sys_prompt = BASE_SYSTEM_PROMPT

    combo_ctx = build_combo_context(combo)
    product_ctx = build_product_context(product)
    profile_ctx = build_profile_context(profile)
    intent_text = f"Intent hiện tại (ước đoán vấn đề sức khỏe): {intent or 'chưa rõ'}."

    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.4,
            messages=[
                {"role": "system", "content": sys_prompt},
                {
                    "role": "system",
                    "content": (
                        "Dữ liệu nội bộ của WELLLAB cho case này:\n"
                        + intent_text
                        + "\n\n[HỒ SƠ KHÁCH HÀNG (nếu có)]: "
                        + profile_ctx
                        + "\n\n[COMBO LIÊN QUAN]:\n"
                        + combo_ctx
                        + "\n\n[SẢN PHẨM LIÊN QUAN]:\n"
                        + product_ctx
                    ),
                },
                {"role": "user", "content": user_text},
            ],
        )
        return (completion.choices[0].message.content or "").strip()
    except Exception as e:
        print("Lỗi gọi OpenAI:", e)
        return "Hiện hệ thống AI đang bận, anh/chị thử lại sau một chút giúp em nhé."


# ========= CÂU CHÀO ĐƠN GIẢN =========
def is_simple_greeting(text: str) -> bool:
    t = text.lower().strip()
    simple = ["chào", "chao", "xin chào", "hi", "hello", "alo", "chao em", "chào em"]
    return any(t == s or t.startswith(s + " ") for s in simple)


def greeting_reply_short() -> str:
    return (
        "Em chào anh/chị 👋\n"
        "Anh/chị cứ mô tả case khách hoặc câu hỏi của mình, em luôn sẵn sàng hỗ trợ ạ. 😊"
    )


# ========= GỬI TIN =========
def send_message(chat_id: int, text: str, keyboard=None):
    try:
        log_event(chat_id, "bot", text, extra={"source": "bot_reply"})
    except Exception as e:
        print("Lỗi log bot:", e)

    payload: dict = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "Markdown",
    }
    if keyboard:
        payload["reply_markup"] = {
            "keyboard": keyboard,
            "resize_keyboard": True,
            "one_time_keyboard": False,
        }

    try:
        requests.post(
            f"{TELEGRAM_API_URL}/sendMessage",
            json=payload,
            timeout=10,
        )
    except Exception as e:
        print("Lỗi gửi message về Telegram:", e)


# ========= ROUTES =========
@app.route("/", methods=["GET"])
def index():
    return "Bot is running.", 200


@app.route("/webhook", methods=["POST"])
def webhook():
    update = request.get_json(force=True, silent=True) or {}
    print("Update:", update)

    message = update.get("message")
    if not message:
        return "no message", 200

    chat_id = message["chat"]["id"]
    text = message.get("text") or ""
    text_stripped = text.strip()

    tg_user = message.get("from") or {}
    user_id = tg_user.get("id", chat_id)
    profile = get_or_create_user_profile(user_id, tg_user)

    log_event(
        user_id,
        "user",
        text_stripped,
        extra={"username": profile.get("username"), "name": profile.get("name")},
    )

    session = get_session(chat_id)

    # ----- LỆNH CƠ BẢN -----
    if text_stripped.startswith("/start"):
        session["mode"] = "tvv"
        session["intent"] = None
        session["profile"] = {}
        session["stage"] = "await_need"
        session["first_issue"] = None
        session["need"] = None
        session["last_combo"] = None
        session["last_product"] = None

        send_message(
            chat_id,
            build_welcome_message(),
            keyboard=get_main_menu_keyboard(),
        )
        return "ok", 200

    if text_stripped.lower() == "/tvv":
        session["mode"] = "tvv"
        send_message(
            chat_id,
            "Đã chuyển sang *chế độ TƯ VẤN VIÊN* (training nội bộ). Anh/chị mô tả case khách hoặc hỏi về combo/sản phẩm nhé.",
            keyboard=get_main_menu_keyboard(),
        )
        return "ok", 200

    if text_stripped.lower() == "/kh":
        session["mode"] = "customer"
        send_message(
            chat_id,
            "Đã chuyển tạm sang *chế độ giả lập khách hàng* để anh/chị luyện hội thoại. Anh/chị nhập thử lời của khách, em sẽ trả lời như tư vấn viên.",
            keyboard=get_main_menu_keyboard(),
        )
        return "ok", 200

    # ----- MENU NHANH -----
    if "Phân tích case khách" in text_stripped:
        session["need"] = "health"
        session["stage"] = "start"
        session["intent"] = None
        session["first_issue"] = None
        ask = (
            "Anh/chị mô tả giúp em case khách nhé:\n"
            "- Vấn đề chính KH đang gặp là gì (mất ngủ, đau đầu, đau dạ dày...)?\n"
            "- Tuổi, giới tính, bệnh nền, thuốc đang dùng (nếu biết)?\n"
            "- Mục tiêu KH: giảm triệu chứng, phòng tái phát hay nâng sức khỏe tổng thể?\n"
            "Sau đó em sẽ gợi ý câu hỏi khai thác thêm + combo/phác đồ phù hợp."
        )
        send_message(chat_id, ask)
        touch_user_stats(profile, need="health", intent=None)
        return "ok", 200

    if "Hỏi combo / sản phẩm" in text_stripped:
        session["need"] = "product"
        session["stage"] = "product_clarify"
        session["intent"] = None
        session["first_issue"] = None
        ask = (
            "Anh/chị cho em biết muốn hỏi về *combo* hay *sản phẩm lẻ* nhé:\n"
            "- Gõ *tên combo* hoặc *bộ sản phẩm cho vấn đề ...* (vd: combo cho mất ngủ, combo gan mật...)\n"
            "- Hoặc gõ *tên/mã sản phẩm* để xem thông tin chi tiết + link.\n"
            "Nếu là case thực tế, anh/chị mô tả thêm tình trạng KH để em gợi ý cách tư vấn luôn ạ."
        )
        send_message(chat_id, ask)
        touch_user_stats(profile, need="product", intent=None)
        return "ok", 200

    if "Chính sách & xử lý từ chối" in text_stripped:
        session["need"] = "policy"
        session["stage"] = "start"
        session["intent"] = None
        session["first_issue"] = None
        ask = (
            "Anh/chị muốn hỏi về *chính sách* hay *xử lý từ chối* nào của KH ạ?\n"
            "- Ví dụ: phí ship, đổi trả, chương trình khuyến mãi...\n"
            "- Hoặc từ chối kiểu: 'đắt quá', 'anh đang uống thuốc bác sĩ', 'anh không tin TPCN'...\n"
            "Anh/chị cứ gõ nguyên văn câu KH nói, em sẽ gợi ý cách xử lý."
        )
        send_message(chat_id, ask)
        touch_user_stats(profile, need="policy", intent=None)
        return "ok", 200

    # ----- CHÀO HỎI -----
    if is_simple_greeting(text_stripped):
        if not session.get("need"):
            session["stage"] = "await_need"
            send_message(
                chat_id,
                build_welcome_message(),
                keyboard=get_main_menu_keyboard(),
            )
        else:
            send_message(chat_id, greeting_reply_short())
        return "ok", 200

    # ----- NÓI “KHÔNG CÓ VẤN ĐỀ SỨC KHOẺ” -----
    if is_no_health_intent(text_stripped):
        session["need"] = "other"
        session["intent"] = None
        session["stage"] = "start"
        session["first_issue"] = None

        reply = (
            "Ok anh/chị 😊\n"
            "Nếu không có case sức khỏe cụ thể, anh/chị vẫn có thể:\n"
            "- Hỏi về sản phẩm/combo để nắm rõ thông tin.\n"
            "- Hỏi kịch bản chăm sóc, follow-up, chốt đơn.\n"
            "- Hỏi về chính sách, chương trình, xử lý từ chối.\n\n"
            "Anh/chị muốn bắt đầu từ phần nào ạ?"
        )
        send_message(chat_id, reply)
        touch_user_stats(profile, need="other", intent=None)
        return "ok", 200

    # ----- CẬP NHẬT HỒ SƠ CƠ BẢN -----
    prof_update = extract_profile(text_stripped)
    if prof_update:
        session["profile"] = {**session.get("profile", {}), **prof_update}

    # ----- FAQ / OBJECTION (KHÔNG TỐN TOKEN) -----
    faq_answer = try_answer_faq(text_stripped)
    if faq_answer:
        send_message(chat_id, faq_answer)
        need_auto = session.get("need") or detect_need(text_stripped)
        session["need"] = need_auto
        touch_user_stats(profile, need=need_auto, intent=None)
        return "ok", 200

    obj_answer = try_answer_objection(text_stripped)
    if obj_answer:
        send_message(chat_id, obj_answer)
        need_auto = session.get("need") or detect_need(text_stripped)
        session["need"] = need_auto
        touch_user_stats(profile, need=need_auto, intent=None)
        return "ok", 200

    # ====== XÁC ĐỊNH NEED ======
    lower = text_stripped.lower()
    explicit_need = None

    if any(kw in lower for kw in ["sản phẩm", "san pham", "combo", "liệu trình", "lieu trinh"]):
        explicit_need = "product"

    if any(
        kw in lower
        for kw in [
            "chính sách",
            "mua hàng",
            "dat hang",
            "đặt hàng",
            "ship",
            "giao hàng",
            "thanh toán",
            "thanh toan",
            "đổi trả",
            "doi tra",
            "bảo hành",
            "bao hanh",
        ]
    ):
        explicit_need = "policy"

    if any(
        kw in lower
        for kw in [
            "sức khỏe",
            "suc khoe",
            "đau ",
            "bị đau",
            "benh",
            "bệnh",
            "triệu chứng",
            "huyết áp",
            "tieu duong",
            "tiểu đường",
            "mỡ máu",
            "gan",
            "thận",
            "da cơ địa",
            "vảy nến",
            "mat ngu",
            "mất ngủ",
            "ho",
            "khó thở",
            "kho tho",
            "viem",
        ]
    ):
        explicit_need = explicit_need or "health"

    if explicit_need:
        session["need"] = explicit_need
        if session.get("stage") == "await_need":
            session["stage"] = "start"
    elif not session.get("need") or session.get("stage") == "await_need":
        session["need"] = detect_need(text_stripped)
        session["stage"] = "start"

    need = session.get("need") or "other"

    # ====== NHÁNH CHÍNH SÁCH ======
    if need == "policy":
        faq_answer = try_answer_faq(text_stripped)
        if faq_answer:
            send_message(chat_id, faq_answer)
            touch_user_stats(profile, need=need, intent=None)
            return "ok", 200

        reply = call_openai_for_answer(
            "Đây là tư vấn viên đang hỏi về CHÍNH SÁCH hoặc CÁCH XỬ LÝ TỪ CHỐI để tư vấn lại cho khách.\n"
            "Hãy trả lời như đang training nội bộ: giải thích rõ, sau đó gợi ý 2–3 câu có thể nói với khách.\n\n"
            f"Câu hỏi/tình huống của tư vấn viên: {text_stripped}",
            session,
            combo=None,
            product=None,
        )
        send_message(chat_id, reply)
        touch_user_stats(profile, need=need, intent=None)
        return "ok", 200

    # ====== NHÁNH SẢN PHẨM / COMBO ======
    if need == "product":
        last_combo = session.get("last_combo")
        last_product = session.get("last_product")

        # 0. Hỏi link của sản phẩm gần nhất
        if last_product and any(
            kw in lower for kw in ["link", "đường link", "duong link", "url", "website", "trang web"]
        ):
            link = last_product.get("link", "")
            base = format_product_for_tvv(last_product)
            if not link:
                base += "\n\n(Sản phẩm này hiện chưa có link trong dữ liệu nội bộ.)"
            send_message(chat_id, base)
            touch_user_stats(profile, need=need, intent=session.get("intent"))
            return "ok", 200

        # 1. Hỏi link của combo gần nhất
        if last_combo and any(
            kw in lower for kw in ["link", "đường link", "duong link", "url", "website", "trang web"]
        ):
            combo_text = format_combo_for_tvv(last_combo)
            send_message(chat_id, combo_text)
            touch_user_stats(profile, need=need, intent=session.get("intent"))
            return "ok", 200

        # 2. TVV gõ tên / mã sản phẩm cụ thể
        prod_matches = search_product_by_text(text_stripped, top_k=1)
        if prod_matches:
            prod = prod_matches[0]
            session["last_product"] = prod
            session["intent"] = "product_info"

            info_block = format_product_for_tvv(prod)
            coach_block = call_openai_for_answer(
                "Tư vấn viên đang hỏi về *một sản phẩm cụ thể* dưới đây.\n"
                "Hãy hướng dẫn cách GIẢI THÍCH đơn giản cho khách (đối tượng dùng, lợi ích chính, cách dùng), "
                "và gợi ý 1–2 câu chốt đơn mềm, không lặp lại toàn bộ thông tin chi tiết y nguyên.\n",
                session,
                combo=None,
                product=prod,
            )
            final_reply = info_block + "\n\n---\n" + coach_block
            send_message(chat_id, final_reply)
            touch_user_stats(profile, need=need, intent=session.get("intent"))
            return "ok", 200

        # 3. TVV gõ tên combo / bộ sản phẩm cụ thể
        matches = search_combo_by_text(text_stripped, top_k=1)
        if matches:
            combo = matches[0]
            session["last_combo"] = combo
            if not session.get("intent"):
                session["intent"] = "product_combo"

            combo_info = format_combo_for_tvv(combo)
            coach_block = call_openai_for_answer(
                "Tư vấn viên đang hỏi về *một combo/bộ sản phẩm cụ thể*.\n"
                "Hãy hướng dẫn cách giải thích cho khách: vấn đề sức khoẻ nào phù hợp, "
                "ưu điểm của combo, cách dùng tổng quát, và gợi ý 1–2 câu chốt.\n",
                session,
                combo=combo,
                product=None,
            )
            final_reply = combo_info + "\n\n---\n" + coach_block
            send_message(chat_id, final_reply)
            touch_user_stats(profile, need=need, intent=session.get("intent"))
            return "ok", 200

        # 4. Không nhận diện được -> hỏi rõ thêm
        session["stage"] = "product_clarify"
        ask = (
            "Để em hỗ trợ đúng hơn, anh/chị cho em biết:\n"
            "- Anh/chị đang cần *combo/bộ sản phẩm* cho VẤN ĐỀ SỨC KHỎE nào của khách (vd: đau đầu, mất ngủ...)?\n"
            "- Hay anh/chị đang cần *thông tin chi tiết* của *1 sản phẩm lẻ* (tên/mã sản phẩm)?\n"
            "Anh/chị có thể gõ: 'combo cho đau đầu', 'bộ cho mất ngủ', hoặc tên/mã sản phẩm cụ thể."
        )
        send_message(chat_id, ask)
        touch_user_stats(profile, need=need, intent=None)
        return "ok", 200

    # ====== OTHER (CHƯA RÕ) ======
    if need == "other" and not detect_intent_from_text(text_stripped):
        reply = (
            "Anh/chị đang muốn:\n"
            "- Phân tích case khách (triệu chứng, bệnh nền...)?\n"
            "- Hỏi combo/sản phẩm cụ thể?\n"
            "- Hay hỏi về chính sách / xử lý từ chối?\n"
            "Anh/chị nói rõ giúp em để em hỗ trợ trúng ý hơn ạ."
        )
        send_message(chat_id, reply)
        touch_user_stats(profile, need=need, intent=None)
        return "ok", 200

    # ====== FLOW SỨC KHOẺ (CASE KHÁCH) ======
    if need == "health":
        new_intent = detect_intent_from_text(text_stripped)
        if new_intent:
            session["intent"] = new_intent

        intent = session.get("intent")
        stage = session.get("stage", "start")

        touch_user_stats(profile, need=need, intent=intent)

        # 1. ĐANG CLARIFY -> coi đây là thông tin bổ sung, tư vấn combo
        if stage == "clarify":
            issue = session.get("first_issue") or ""
            if not issue:
                session["first_issue"] = text_stripped
                issue = text_stripped

            combined_user_text = (
                "Tư vấn viên mô tả case khách như sau.\n"
                "Mô tả ban đầu: " + issue + "\n\n"
                "Thông tin bổ sung: " + text_stripped + "\n\n"
                "Hãy giúp tư vấn viên: (1) tóm tắt lại case, "
                "(2) gợi ý thêm vài câu hỏi nếu cần, "
                "(3) gợi ý combo + cách giải thích cho khách, "
                "(4) gợi ý 1–2 câu chốt.\n"
                "Nhớ: combo được chọn phải đúng với vấn đề sức khoẻ, và tư vấn viên cần có link từng sản phẩm trong combo (đã có sẵn trong dữ liệu)."
            )

            combo = choose_combo(intent)
            session["last_combo"] = combo
            session["stage"] = "advise"

            combo_info = format_combo_for_tvv(combo) if combo else "Hiện chưa map được combo rõ ràng cho case này."
            coach_block = call_openai_for_answer(combined_user_text, session, combo=combo, product=None)
            final_reply = combo_info + "\n\n---\n" + coach_block
            send_message(chat_id, final_reply)
            return "ok", 200

        # 2. CHƯA CÓ INTENT RÕ
        if not intent:
            question = get_clarify_question(None)
            session["stage"] = "clarify"
            if not session.get("first_issue"):
                session["first_issue"] = text_stripped
            send_message(chat_id, question)
            return "ok", 200

        # 3. CÓ INTENT, ĐANG Ở START
        if stage in ("start", None):
            session["first_issue"] = text_stripped
            session["stage"] = "clarify"
            question = get_clarify_question(intent)
            send_message(chat_id, question)
            return "ok", 200

        # 4. GIAI ĐOẠN ADVISE -> câu hỏi bổ sung sau khi đã tư vấn combo
        if stage == "advise":
            combo = choose_combo(intent)
            session["last_combo"] = combo
            coach_block = call_openai_for_answer(
                "Tư vấn viên đang hỏi thêm về cùng 1 case khách ở trên. "
                "Hãy tiếp tục hỗ trợ đào sâu (xử lý thắc mắc, từ chối, nhắc lại cách dùng, follow-up...).\n\n"
                "Câu hỏi bổ sung của tư vấn viên: " + text_stripped,
                session,
                combo=combo,
                product=None,
            )
            # Ở giai đoạn này không cần lặp lại full combo, chỉ cần câu trả lời coaching
            send_message(chat_id, coach_block)
            return "ok", 200

        # Fallback trong health
        combo = choose_combo(intent)
        session["last_combo"] = combo
        combo_info = format_combo_for_tvv(combo) if combo else ""
        coach_block = call_openai_for_answer(text_stripped, session, combo=combo, product=None)
        final_reply = (combo_info + "\n\n---\n" + coach_block) if combo_info else coach_block
        send_message(chat_id, final_reply)
        return "ok", 200

    # ====== FALLBACK CHUNG ======
    intent = session.get("intent")
    combo = choose_combo(intent)
    session["last_combo"] = combo
    reply = call_openai_for_answer(text_stripped, session, combo=combo, product=None)
    send_message(chat_id, reply)
    touch_user_stats(profile, need=need, intent=intent)
    return "ok", 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
