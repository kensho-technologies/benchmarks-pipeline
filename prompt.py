# Copyright 2024-present Kensho Technologies, LLC.
import re
import json
import random

from dataclasses import dataclass, fields
from typing import List, Tuple
from python import exec_python

TASKS = ["CodeTAT-QA", "TAT-QA", "CodeFinQA", "FinKnow", "FinCode", "ConvFinQA"]

LETTERS = "ABCD"


@dataclass
class Output:
    question_id: str
    answer: float


@dataclass
class Question:
    id: str
    task: str
    question: str

    def __str__(self):
        return self.question

    @classmethod
    def from_json(cls, json_data):
        return cls(**json_data)

    def to_json(self):
        return {f.name: getattr(self, f.name) for f in fields(self.__class__)}


def multiple_choice_string_format(question, options):
    options = [LETTERS[idx] + ". " + o for (idx, o) in enumerate(options)]
    options_str = "\n".join(options)
    return f"Question: {question}\n" + options_str


@dataclass
class MultipleChoiceQuestion(Question):
    options: List[str]

    def __str__(self):
        return multiple_choice_string_format(self.question, self.options)


def context_string_format(context, question, context_type):
    return f"Context:\n{context}\n\nQuestion: {question}\n"


@dataclass
class ContextualQuestion(Question):
    context: str
    context_type: str

    def __str__(self):
        return context_string_format(self.context, self.question, self.context_type)


def parse_question_json(data):
    if "options" in data.keys():
        return MultipleChoiceQuestion(**data)
    elif "context" in data.keys():
        return ContextualQuestion(**data)
    else:
        return Question(**data)


def load_data(path):
    data = json.load(open(path, "r"))
    data = [parse_question_json(d) for d in data]
    random.shuffle(data)
    return data


def create_prompt(question, samples=[], system_message=None):
    messages = []

    if system_message is not None:
        messages.append({"role": "system", "content": system_message})

    for sample_question, sample_answer in samples:
        messages.append({"role": "user", "content": str(sample_question)})
        messages.append({"role": "assistant", "content": str(sample_answer)})

    messages.append({"role": "user", "content": str(question)})
    return messages


def load_examples(json_file):
    data = json.load(open(json_file, "r"))

    system_message = data["system_message"]
    samples = [
        (parse_question_json(d["question"]), d["answer"]) for d in data["samples"]
    ]
    return system_message, samples


def brace_extract(output: str) -> Tuple[bool, float]:
    outs = re.findall(r"\[\[(.*)\]\]", output)

    if len(outs) != 1:
        return False, -1.0

    value = outs[0]

    # Remove some common leftover strings
    for s in [",", "$", "%"]:
        value = value.replace(s, "")

    if value in LETTERS:
        choice = LETTERS.index(value)
        return True, float(choice)

    elif is_float(value):
        return True, float(value)

    else:
        return False, -1.0


def python_extract(output: str) -> Tuple[bool, float]:
    pattern = r"^```(?:\w+)?\s*\n(.*?)(?=^```)```"
    code = re.findall(pattern, output, re.DOTALL | re.MULTILINE)

    if len(code) != 1:
        return False, -1.0

    code = code[0]
    return_dict = exec_python(code)
    return_val = return_dict["return_val"]

    if return_val == None or not is_float(return_val):
        print("Failed to parse python code due to: ", return_dict["failure_reason"])
        return False, -1.0
    else:
        return True, return_val


class PromptCreator:
    def __init__(self, prompt_json_map):
        self.prompt_map = {}
        for t in TASKS:
            self.prompt_map[t] = load_examples(prompt_json_map[t])

    def create(self, question):
        system_message, samples = self.prompt_map[question.task]

        messages = create_prompt(
            question,
            samples=samples,
            system_message=system_message,
        )
        return messages

    def parse_output(self, output: str) -> Tuple[bool, float]:
        if "```" in output:
            return python_extract(output)
        else:
            return brace_extract(output)


def is_float(value: str):
    try:
        value = float(value)
        return True
    except ValueError:
        return False
