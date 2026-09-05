from langchain_openai import ChatOpenAI

LLM_API_URL_CORE = "http://10.0.99.116:8070/v1"
MODEL_NAME_CORE = "Qwen/Qwen3.5-35B-A3B"


def load_llm():
    return ChatOpenAI(
        model=MODEL_NAME_CORE,
        base_url=LLM_API_URL_CORE,
        api_key="sk-no-key-required",
        temperature=0,
        max_tokens=2048,
        top_p=0.8,
        presence_penalty=1.5,
        extra_body={
            "top_k": 20,
            "chat_template_kwargs": {
                "enable_thinking": False
            }
        }
    )