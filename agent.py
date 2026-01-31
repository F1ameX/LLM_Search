import json
import os
from dotenv import load_dotenv
from tools.search import web_search
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage,SystemMessage, ToolMessage
load_dotenv()

model = init_chat_model(
    model="meta-llama/llama-3.3-70b-instruct:free",
    model_provider="openai",
    api_key=os.environ["API_KEY"],
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
    streaming=True,
)

tools = [web_search]
tools_by_name = {t.name: t for t in tools}
model_with_tools = model.bind_tools(tools)

with open("prompt.txt", "r", encoding="utf-8") as f:
    prompt = f.read()

messages = [SystemMessage(content=prompt)]

def _shrink_tool_result(result, max_sources=4, max_chars_per_source=3000):
    ok = [x for x in result if (not x.get("error")) and (x.get("char_count", 0) > 800)]
    ok.sort(key=lambda x: x.get("char_count", 0), reverse=True)
    ok = ok[:max_sources]
    for x in ok:
        x["text"] = (x.get("text") or "")[:max_chars_per_source]
        x.pop("excerpt", None)
    return ok

while True:
    user_input = input("\nYou> ").strip()
    if user_input.lower() in {"exit", "quit", "q"}:
        break

    messages.append(HumanMessage(content=user_input))

    max_steps = 5
    for _ in range(max_steps):

        merged = None
        print("\nAssistant> ", end="", flush=True)

        for chunk in model_with_tools.stream(messages):
            if chunk.content:
                print(chunk.content, end="", flush=True)
            merged = chunk if merged is None else (merged + chunk)

        print()

        if merged is None:
            break

        try:
            resp_msg = merged.to_message()
        except Exception:
            resp_msg = merged

        messages.append(resp_msg)

        tool_calls = getattr(resp_msg, "tool_calls", None) or []
        if not tool_calls:
            break
        for tc in tool_calls:
            name = tc["name"]
            args = tc.get("args", {}) or {}
            tool = tools_by_name[name]

            result = tool.invoke(args)

            if name == "web_search" and isinstance(result, list):
                result = _shrink_tool_result(result)

            messages.append(
                ToolMessage(
                    content=json.dumps(result, ensure_ascii=False),
                    tool_call_id=tc["id"],
                )
            )