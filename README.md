# PyGPT-J
* Python bindings for GPT-J  [ggml](https://github.com/ggerganov/ggml) language models.
* Almost the same API as [pyllamacpp](https://github.com/abdeladim-s/pyllamacpp).

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![PyPi version](https://badgen.net/pypi/v/pygptj)](https://pypi.org/project/pygptj/)

# Table of contents
<!-- TOC -->
* [Installation](#installation)
* [CLI](#cli)
* [Tutorial](#tutorial)
    * [Quick start](#quick-start)
    * [Interactive Dialogue](#interactive-dialogue)
    * [Attribute a persona to the language model](#attribute-a-persona-to-the-language-model)
* [API reference](#api-reference)
* [License](#license)
<!-- TOC -->

# Installation
1. The easy way is to use the prebuilt wheels
```bash
pip install pygptj
```

2. Build from source:

```shell
git clone git+https://github.com/abdeladim-s/pygptj.git
```

# CLI 

You can run the following simple command line interface to test the package once it is installed:

```shell
pygtj path/to/ggml/model
```

# Tutorial

### Quick start

```python
from pygptj.model import Model

model = Model(model_path='path/to/gptj/ggml/model')
for token in model.generate("Tell me a joke ?"):
    print(token, end='', flush=True)
```

### Interactive Dialogue
You can set up an interactive dialogue by simply keeping the `model` variable alive:

```python
from pygptj.model import Model

model = Model(model_path='/path/to/ggml/model')
while True:
    try:
        prompt = input("You: ", flush=True)
        if prompt == '':
            continue
        print(f"AI:", end='')
        for token in model.generate(prompt):
            print(f"{token}", end='', flush=True)
        print()
    except KeyboardInterrupt:
        break
```
### Attribute a persona to the language model

The following is an example showing how to _"attribute a persona to the language model"_ :

```python
from pygptj.model import Model

prompt_context = """Act as Bob. Bob is helpful, kind, honest,
and never fails to answer the User's requests immediately and with precision. 

User: Nice to meet you Bob!
Bob: Welcome! I'm here to assist you with anything you need. What can I do for you today?
"""

prompt_prefix = "\nUser:"
prompt_suffix = "\nBob:"

model = Model(model_path='/path/to/ggml/model',
              prompt_context=prompt_context,
              prompt_prefix=prompt_prefix,
              prompt_suffix=prompt_suffix)

while True:
  try:
    prompt = input("User: ")
    if prompt == '':
      continue
    print(f"Bob: ", end='')
    for token in model.generate(prompt, antiprompt='User:'):
      print(f"{token}", end='', flush=True)
      print()
  except KeyboardInterrupt:
    break
```

[//]: # (* You can always refer to the [short documentation]&#40;https://nomic-ai.github.io/pyllamacpp/&#41; for more details.)


# API reference
You can check the [API reference documentation](https://abdeladim-s.github.io/pygptj/) for more details.

# License

This project is licensed under the MIT  [License](./LICENSE).

