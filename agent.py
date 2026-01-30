import json
import os
from tools.search import web_search
from dotenv import load_dotenv

from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, SystemMessage, ToolMessage

load_dotenv()

model = init_chat_model(
    model="meta-llama/llama-3.3-70b-instruct:free",
    model_provider="openai",
    api_key=os.environ["API_KEY"],
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
)

tools = [web_search]
tools_by_name = {t.name: t for t in tools}
model_with_tools = model.bind_tools(tools)

messages = [
    SystemMessage(content="You are a helpful assistant. Use tools when needed."),
    HumanMessage(content="Найди в интернете как правильно подобрать велосипед и кратко суммируй по пунктам."),
]

max_steps = 5
for _ in range(max_steps):
    resp = model_with_tools.invoke(messages)
    messages.append(resp)

    tool_calls = getattr(resp, "tool_calls", None) or []
    if not tool_calls:
        break

    for tc in tool_calls:
        name = tc["name"]
        args = tc.get("args", {})
        tool = tools_by_name[name]

        result = tool.invoke(args) 
        messages.append(
            ToolMessage(
                content=json.dumps(result, ensure_ascii=False),
                tool_call_id=tc["id"],
            )
        )

print(messages[-1].content)