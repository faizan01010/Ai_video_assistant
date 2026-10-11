#Actionableitems , decision , questions
 
import json
import os
import re
import time
 
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
 
 
def get_llm():
    return ChatGroq(model="openai/gpt-oss-120b", groq_api_key=os.getenv("GROQ_API_KEY"), temperature=0.2)
 
 
def build_chain(system_prompt: str):
    llm = get_llm()
    return (
        RunnablePassthrough() | RunnableLambda(lambda x: {"text": x}) | ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{text}"),
        ]) | llm | StrOutputParser()
    )
 
 
def _invoke_with_retry(chain, payload, retries: int = 5):
    """Retry on Groq rate limits (429), waiting as long as the error message asks."""
    for attempt in range(retries):
        try:
            return chain.invoke(payload)
        except Exception as e:
            msg = str(e)
            if "429" not in msg and "rate_limit" not in msg:
                raise
            m = re.search(r"try again in ([\d.]+)\s*(ms|s)", msg)
            if m:
                wait = float(m.group(1)) / (1000 if m.group(2) == "ms" else 1) + 1
            else:
                wait = 2 ** attempt * 2
            print(f"Rate limited, waiting {wait:.1f}s (attempt {attempt + 1}/{retries})...")
            time.sleep(wait)
    raise RuntimeError("Groq rate limit: retries exhausted")
 
 
# ─── One combined call (3x fewer tokens than three separate calls) ───────────────
 
# NOTE: braces are doubled because this string goes through ChatPromptTemplate.
_COMBINED_PROMPT = (
    "You are an expert meeting analyst. Read the meeting transcript and return ONLY a "
    "valid JSON object, with no markdown fences and no commentary, using exactly these keys:\n"
    '{{"action_items": [{{"task": "...", "owner": "...", "deadline": "..."}}], '
    '"key_decisions": ["..."], '
    '"open_questions": ["..."]}}\n\n'
    "Rules:\n"
    "- action_items: every task someone committed to. Use 'Not specified' when the owner "
    "or the deadline is not mentioned.\n"
    "- key_decisions: every decision that was made.\n"
    "- open_questions: unresolved questions or topics needing follow-up.\n"
    "- Use an empty list when nothing is found. Never invent information."
)
 
 
def _parse_json(raw: str) -> dict:
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.I)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.S)
        if m:
            return json.loads(m.group(0))
        raise
 
 
def _fmt_actions(items) -> str:
    if not items:
        return "No action items found."
    lines = []
    for i, it in enumerate(items, 1):
        if isinstance(it, dict):
            task = str(it.get("task", "")).strip()
            owner = str(it.get("owner") or "Not specified").strip()
            deadline = str(it.get("deadline") or "Not specified").strip()
            lines.append(f"{i}. {task} (Owner: {owner}, Deadline: {deadline})")
        else:
            lines.append(f"{i}. {it}")
    return "\n".join(lines)
 
 
def _fmt_list(items, empty_msg: str) -> str:
    if not items:
        return empty_msg
    return "\n".join(f"{i}. {str(it).strip()}" for i, it in enumerate(items, 1))
 
 
def extract_all(transcript: str):
    """Returns (action_items, key_decisions, open_questions) as numbered-list strings."""
    chain = build_chain(_COMBINED_PROMPT)
    raw = _invoke_with_retry(chain, transcript)
    try:
        data = _parse_json(raw)
        return (
            _fmt_actions(data.get("action_items")),
            _fmt_list(data.get("key_decisions"), "No key decisions found."),
            _fmt_list(data.get("open_questions"), "No open questions found."),
        )
    except (ValueError, AttributeError, TypeError) as e:
        # Model returned something that is not valid JSON -> fall back to the 3 separate calls
        print(f"Combined extraction could not be parsed ({e}); using 3 separate calls.")
        return (
            extract_action_items(transcript),
            extract_key_decisions(transcript),
            extract_questions(transcript),
        )
 
 
# ─── Original single-purpose functions (kept for compatibility / fallback) ──────
 
def extract_action_items(transcript: str) -> str:
    chain = build_chain(
        "You are an expert meeting analyst. From the meeting transcript, "
        "extract all action items. For each provide:\n"
        "- Task description\n"
        "- Owner (who is responsible)\n"
        "- Deadline (if mentioned, else write 'Not specified')\n\n"
        "Format as a numbered list. If none found say 'No action items found.'"
    )
 
    return _invoke_with_retry(chain, transcript)
 
 
def extract_key_decisions(transcript: str) -> str:
    chain = build_chain(
        "You are an expert meeting analyst. From the meeting transcript, "
        "extract all key decisions made. Format as a numbered list. "
        "If none found say 'No key decisions found.'"
    )
    return _invoke_with_retry(chain, transcript)
 
 
def extract_questions(transcript: str) -> str:
    chain = build_chain(
        "From the meeting transcript, extract all unresolved questions "
        "or topics needing follow-up. Format as a numbered list. "
        "If none found say 'No open questions found.'"
    )
    return _invoke_with_retry(chain, transcript)