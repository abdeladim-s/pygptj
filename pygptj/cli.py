#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A quick and simple command line interface
"""

import argparse
import importlib.metadata
import logging


__version__ = importlib.metadata.version('pygptj')

__header__ = f"""

██████╗ ██╗   ██╗ ██████╗ ██████╗ ████████╗        ██╗
██╔══██╗╚██╗ ██╔╝██╔════╝ ██╔══██╗╚══██╔══╝        ██║
██████╔╝ ╚████╔╝ ██║  ███╗██████╔╝   ██║█████╗     ██║
██╔═══╝   ╚██╔╝  ██║   ██║██╔═══╝    ██║╚════╝██   ██║
██║        ██║   ╚██████╔╝██║        ██║      ╚█████╔╝
╚═╝        ╚═╝    ╚═════╝ ╚═╝        ╚═╝       ╚════╝ 
                                                                                                 
PyGPT-J
A simple Command Line Interface to test the package
Version: {__version__}               
=========================================================================================
"""

from pygptj.model import Model

prompt_context = """ Act as GPTJ. GPTJ is helpful, kind, honest, good at writing, and never fails to answer the User's requests immediately and with precision. 

User: Nice to meet you GPTJ!
GPTJ: Welcome! I'm here to assist you with anything you need. What can I do for you today?
"""

prompt_prefix = "\nUser:"
prompt_suffix = "\nGPTJ:"

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def run(args):
    print(f"[+] Running model `{args.model}`")
    model = Model(model_path=args.model,
                  prompt_context=prompt_context,
                  prompt_prefix=prompt_prefix,
                  prompt_suffix=prompt_suffix,
                  log_level=logging.ERROR)
    print("...")
    print("[+] Press Ctrl+C to Stop ... ")
    print("...")

    stop_word = prompt_prefix.strip()
    while True:
        try:
            prompt = input("You: ")
            if prompt == '':
                continue
            print(f"{bcolors.OKCYAN}GPTJ: {bcolors.ENDC}", end='', flush=True)
            for token in model.generate(prompt, antiprompt=prompt_prefix.strip()):
                print(f"{bcolors.OKCYAN}{token}{bcolors.ENDC}", end='', flush=True)
            print()
        except KeyboardInterrupt:
            break


def main():
    print(__header__)

    parser = argparse.ArgumentParser(description="", allow_abbrev=True)
    # Positional args
    parser.add_argument('model', type=str, help="The path of the model file")

    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
