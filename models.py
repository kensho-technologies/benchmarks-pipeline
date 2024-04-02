# Copyright 2024-present Kensho Technologies, LLC.
import os
import boto3
import torch
from openai import OpenAI
from dataclasses import dataclass

# import google.generativeai as genai
from transformers import AutoModelForCausalLM, AutoTokenizer

import json
from retry import retry
import logging

logger = logging.getLogger(__name__)


class Model:
    def __call__(self, data) -> str:
        ...


@dataclass
class OpenAIChatModel(Model):
    def __init__(self, model, model_kwargs=None):
        self.client = OpenAI()
        self.model = model

        self.model_kwargs = model_kwargs
        if self.model_kwargs is None:
            self.model_kwargs = {}

    @retry(delay=1, logger=logger, tries=5)
    def __call__(self, messages) -> str:
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **self.model_kwargs,
        )

        return completion.choices[0].message.content


@dataclass
class OpenAIAssistantModel(Model):
    def __init__(self, model, tools=None):
        self.client = OpenAI()
        self.model = model
        self.tools = tools

    @retry(delay=1, logger=logger, tries=5)
    def __call__(self, messages) -> str:
        instruction = None
        if messages[0]["role"] == "system":
            instruction = messages[0]["content"]
            messages = messages[1:]

        assistant = self.client.beta.assistants.create(
            model=self.model,
            instructions=instruction,
            tools=self.tools,
        )

        thread = self.client.beta.threads.create()

        # Few shot is not allowed?
        for m in messages[-1:]:
            message = self.client.beta.threads.messages.create(thread_id=thread.id, **m)

        run = self.client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=assistant.id,
        )

        while run.status != "completed":
            run = self.client.beta.threads.runs.retrieve(
                thread_id=thread.id, run_id=run.id
            )

        messages = self.client.beta.threads.messages.list(thread_id=thread.id)
        return messages.data[0].content[0].text.value


@dataclass
class BedRockModel(Model):
    def __init__(self, model_id: str, max_tokens: int = 1000):
        self.bedrock_runtime = boto3.client(
            service_name="bedrock-runtime", region_name="us-east-1"
        )
        self.anthropic_version = "bedrock-2023-05-31"

        self.model_id = model_id
        self.max_tokens = max_tokens

    @retry(delay=1, logger=logger, tries=5)
    def __call__(self, messages) -> str:
        instruction = None
        if messages[0]["role"] == "system":
            instruction = messages[0]["content"]
            messages = messages[1:]

        body = json.dumps(
            {
                "anthropic_version": self.anthropic_version,
                "max_tokens": self.max_tokens,
                "system": instruction,
                "messages": messages,
            }
        )

        response = self.bedrock_runtime.invoke_model(body=body, modelId=self.model_id)
        response_body = json.loads(response.get("body").read())
        return response_body["content"][0]["text"]


class HFModel(Model):
    def __init__(
        self,
        model_name_or_path,
        device_map=0,
        generate_until=None,
        model_kwargs=None,
        generation_kwargs=None,
    ):
        self.model_kwargs = model_kwargs
        if self.model_kwargs is None:
            self.model_kwargs = {}

        self.generation_kwargs = generation_kwargs
        if self.generation_kwargs is None:
            self.generation_kwargs = {}

        if torch.cuda.is_available():
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path, device_map=device_map, **self.model_kwargs
            )
        else:
            # This will not work for larger models.
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path, **self.model_kwargs
            )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model.config.pad_token_id = (
            self.model.config.eos_token_id
        ) = self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.stop_token = None
        if generate_until:
            tokens = self.tokenizer.encode("\n" + generate_until)
            stop_token = list(
                filter(lambda t: self.tokenizer.decode(t) == generate_until, tokens)
            )

            assert len(stop_token) == 1, "Can't parse tokenizer output!"
            self.stop_token = stop_token[0]

    def _render(self, messages):
        string = ""
        for m in messages:
            role = m["role"]
            content = m["content"]
            string += f"{role}:\n{content}\n"

        string += f"assistant:"
        encodeds = self.tokenizer.encode(string, return_tensors="pt")
        return encodeds

    @retry(delay=1, logger=logger, tries=5)
    def __call__(self, messages) -> str:
        model_inputs = self._render(messages)
        if torch.cuda.is_available():
            model_inputs = model_inputs.cuda()
        generated_ids = self.model.generate(
            model_inputs,
            use_cache=True,
            eos_token_id=self.stop_token,
            do_sample=True,
            **self.generation_kwargs,
        )
        output_str = self.tokenizer.decode(generated_ids[0, len(model_inputs[0]) :])
        print(output_str)
        return output_str


class HFChatModel(HFModel):
    def _render(self, messages):
        encodeds = self.tokenizer.apply_chat_template(messages, return_tensors="pt")
        return encodeds


def transform_to_gemini(messages_chatgpt):
    messages_gemini = []
    system_promt = ""
    for message in messages_chatgpt:
        if message["role"] == "system":
            system_promt = message["content"]
        elif message["role"] == "user":
            messages_gemini.append({"role": "user", "parts": [message["content"]]})
        elif message["role"] == "assistant":
            messages_gemini.append({"role": "model", "parts": [message["content"]]})
    if system_promt:
        messages_gemini[0]["parts"].insert(0, f"*{system_promt}*")

    return messages_gemini


@dataclass
class GeminiModel(Model):
    def __init__(self, model_name):
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        self.model = genai.GenerativeModel(model_name)

    @retry(delay=1, logger=logger, tries=5)
    def __call__(self, messages) -> str:
        response = self.model.generate_content(transform_to_gemini(messages))
        return response.text
