# Copyright 2024-present Kensho Technologies, LLC.
from models import (
    OpenAIChatModel,
    OpenAIAssistantModel,
    HFChatModel,
    HFModel,
    BedRockModel,
    GeminiModel,
)

from prompt import PromptCreator

# Tasks: ['CodeTAT-QA', 'TAT-QA', 'CodeFinQA', 'FinKnow', 'FinCode', 'ConvFinQA']

code_prompt_creator = PromptCreator(
    {
        "FinKnow": "prompts/finknow.json",
        "CodeTAT-QA": "prompts/context_code.json",
        "CodeFinQA": "prompts/codefinqa_code.json",
        "FinCode": "prompts/fincode_code.json",
        "ConvFinQA": "prompts/convfinqa.json",
        "TAT-QA": "prompts/tatqa_e.json",
    }
)

cot_prompt_creator = PromptCreator(
    {
        "FinKnow": "prompts/finknow.json",
        "CodeTAT-QA": "prompts/context_cot.json",
        "CodeFinQA": "prompts/codefinqa_cot.json",
        "FinCode": "prompts/fincode_cot.json",
        "ConvFinQA": "prompts/convfinqa.json",
        "TAT-QA": "prompts/tatqa_e.json",
    }
)

# All names must be in the form {unique model name}-{cot|code}. The cache is
# defined by the unique model name

_CONFIG = {
    "gemini-pro-cot": lambda: (GeminiModel("gemini-pro"), cot_prompt_creator),
    "gemini-pro-code": lambda: (GeminiModel("gemini-pro"), code_prompt_creator),
    "claude-3-sonnet-code": lambda: (
        BedRockModel("anthropic.claude-3-sonnet-20240229-v1:0"),
        code_prompt_creator,
    ),
    "claude-3-sonnet-cot": lambda: (
        BedRockModel("anthropic.claude-3-sonnet-20240229-v1:0"),
        cot_prompt_creator,
    ),
    "gpt-4-code": lambda: (OpenAIChatModel("gpt-4"), code_prompt_creator),
    "gpt-4-cot": lambda: (OpenAIChatModel("gpt-4"), cot_prompt_creator),
    "gpt-3.5-code": lambda: (OpenAIChatModel("gpt-3.5-turbo"), code_prompt_creator),
    "gpt-3.5-cot": lambda: (OpenAIChatModel("gpt-3.5-turbo"), cot_prompt_creator),
    # This has a built in code executor, so we only need to give it the cot prompt
    "gpt-4-assist-no-code-cot": lambda: (
        OpenAIAssistantModel("gpt-4-turbo-preview"),
        cot_prompt_creator,
    ),
    "gpt-4-assist-code": lambda: (
        OpenAIAssistantModel(
            "gpt-4-turbo-preview", tools=[{"type": "code_interpreter"}]
        ),
        cot_prompt_creator,
    ),
    "claude-2-code": lambda: (BedRockModel("anthropic.claude-v2"), code_prompt_creator),
    "claude-2-cot": lambda: (BedRockModel("anthropic.claude-v2"), cot_prompt_creator),
    "Mistral-7B-v0.1-cot": lambda: (
        HFModel("mistralai/Mistral-7B-v0.1", generation_kwargs={"max_new_tokens": 256}),
        cot_prompt_creator,
    ),
    # None of the llama 2 models seem to follow the right output format for some reason?
    "llama-2-7b-chat-code": lambda: (
        HFChatModel("meta-llama/Llama-2-7b-chat-hf"),
        code_prompt_creator,
    ),
    "llama-2-7b-chat-cot": lambda: (
        HFChatModel("meta-llama/Llama-2-7b-chat-hf"),
        cot_prompt_creator,
    ),
    "llama-2-13b-chat-code": lambda: (
        HFChatModel("meta-llama/Llama-2-13b-chat-hf"),
        code_prompt_creator,
    ),
    "llama-2-13b-chat-cot": lambda: (
        HFChatModel("meta-llama/Llama-2-13b-chat-hf"),
        cot_prompt_creator,
    ),
    "llama-2-70b-chat-code": lambda: (
        HFChatModel("meta-llama/Llama-2-70b-chat-hf", device_map="auto"),
        code_prompt_creator,
    ),
    "llama-2-70b-chat-cot": lambda: (
        HFChatModel("meta-llama/Llama-2-70b-chat-hf", device_map="auto"),
        cot_prompt_creator,
    ),
    "zepyhr-cot": lambda: (
        HFChatModel(
            "HuggingFaceH4/zephyr-7b-beta",
            device_map="auto",
            generation_kwargs={"max_new_tokens": 2048, "pad_token_id": 0},
        ),
        cot_prompt_creator,
    ),
    "zepyhr-code": lambda: (
        HFChatModel(
            "HuggingFaceH4/zephyr-7b-beta",
            device_map="auto",
            generation_kwargs={"max_new_tokens": 2048, "pad_token_id": 0},
        ),
        code_prompt_creator,
    ),
}


def load_config(name):
    return _CONFIG[name]()


def load_hf_config(
    hugging_face_model_name_or_path,
    prompt_style,
    is_chat_model,
    device_map,
    max_new_tokens,
):
    if is_chat_model:
        m = HFChatModel(
            hugging_face_model_name_or_path,
            device_map=device_map,
            generation_kwargs={
                "max_new_tokens": max_new_tokens,
            },
        )
    else:
        m = HFModel(
            hugging_face_model_name_or_path,
            device_map=device_map,
            generation_kwargs={"max_new_tokens": max_new_tokens},
        )
    if prompt_style == "cot":
        p = cot_prompt_creator
    else:
        p = code_prompt_creator
    return m, p
