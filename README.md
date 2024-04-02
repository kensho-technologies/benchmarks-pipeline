# S&P AI Benchmarks Demo Pipeline

This repo shows how to run models over S&P AI Benchmarks. All the configured models can be seen in `config.py`. It is easy to either add your own models to the config, or run huggingface models using commandline options.

## Setup

Please download the questions from our S&P AI Benchmarks website's [submission page](https://benchmarks.kensho.com) and save them directly within this folder, `benchmarks-pipeline/benchmark_questions.json`.

```
# We recommend using python 3.10.6 with pyenv
pyenv install 3.10.6
pyenv local 3.10.6
virtualenv -p python3.10.6 .benchmarks
source .benchmarks/bin/activate

# Install the requirements in your local environment
pip install -r requirements.txt
```

Hardware Requirements: Most models that can run quickly on CPU will not perform well on this benchmark; we recommend using a system with GPUs. To set the device use the `--device_map` parameter.

## Design Decisions

We provide the prompts we use for evaluation; currently all models use the same prompts for a given question type. We allow models multiple attempts to generate an answer in the expected format. Without this retry step we find that some models are unduly harmed by our answer parsing: they produce the correct answer in the wrong format. Thus, we allow models up to 10 attempts to generate an answer in the expected format. The source code in this repo does this by default, but can be controlled by the `-t, --answer_parsing_tries_alloted` parameter.

## Usage

We provide a number of configurations for both open source and propietary models in `config.py`. If you want to use one of those models, then use the codes listed in `config.py`. You can also configure a huggingface model by the commandline args.
```bash
python main.py -m Mistral-7B-v0.1-cot
# or:
python main.py -n mistralai/Mistral-7B-v0.1 --prompt_style cot --max_new_tokens 12 --answer_parsing_tries_alloted 1
```

The output csv includes columns for the question id and answer with no header. See `results/Mistral-7B-v0.1-cot.csv` for an example output.
```csv
# A snapshot from the example output.
35c06bfe-60a7-47b4-ab82-39e138abd629,13428.0
33c7bd71-e5a3-40dd-8eb0-5000c9353977,-4.5
7b60e737-4f0a-467b-9f73-fa5714d8cdbb,41846.0
0a3f6ada-b8d3-48cc-adb4-270af0e08289,2.0
03999e5f-05ee-4b71-95ad-c5a61aae4858,2.0
```

## Configuring a New Model

If you want to add a new model add to the `_CONFIG` variable in config.py. For instance, the following snippet adds the zephyr model with custom default `max_new_tokens`. You must also select the prompt creator you want to use. This controls the prompts created for each question. We provide two, `code_prompt_creater` and `cot_prompt_creator`.
```python
_CONFIG = {
    ...,
    "example-zepyhr-code": lambda: (
        HFChatModel(
            "HuggingFaceH4/zephyr-7b-beta",
            device_map="auto",
            generation_kwargs={"max_new_tokens": 2048},
        ),
        code_prompt_creator,
    ),
}
```

For this specific model you could have used the commandline directly:
```bash
python main.py -n HuggingFaceH4/zephyr-7b-beta --prompt_style code --max_new_tokens 2048 --device_map auto
```

## Upload

Upload your results to S&P AI Benchmarks! See the page [here at https://benchmarks.kensho.com](https://benchmarks.kensho.com).

## Contact

This repo is meant to serve as a template for further experimentation!

Please reach out to `kensho-benchmarks-demo@kensho.com` with any questions.

Copyright 2024-present Kensho Technologies, LLC.
