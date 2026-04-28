import inspect
from types import ModuleType
from typing import List

import numpy as np
import subprocess
import threading
import atexit
import requests
from subprocess import Popen

from structure.canvas.canvas import Canvas
from dsls.our_dsl.functions import dsl_functions as dsl
from structure.geometry.basic_geometry import Colour
from structure.task.task import Task


class ModelClient:
    def __init__(self, system_content:str = "You are a helpful coding assistant.", grammar_file: str | None = None):
        self.URL = "http://127.0.0.1:8080/v1/chat/completions"
        if grammar_file is not None:
            self.grammar = open(grammar_file).read()
            print('---- Grammar -----')
            print(self.grammar)
            print('---- ------- -----')
        else:
            self.grammar = None

        self.server_process = None
        self.messages = [{'role': 'system', 'content': system_content}]
        self.context = ""
        self.response = ""
        atexit.register(self.kill_llama_server)

    def initialise_llama_server(self):
        if self.server_process is None:
            self.server_process = Popen(r'G:\Code\Repos\Mine\Machine_Learning\ARC\arc_with_llm_with_cfg\src\model_interaction\windows_llm_server_startup.bat',
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT,  # merge stderr into stdout
                                        text=True,
                                        bufsize=1
                                        )

            server_ready = threading.Event()  # signals when the string is found

            def watch_stdout():
                for line in self.server_process.stdout:
                    print(line, end="")  # mirror to REPL
                    if "srv  update_slots: all slots are idle" in line:
                        server_ready.set()

            thread = threading.Thread(target=watch_stdout, daemon=True)
            thread.start()

            server_ready.wait(timeout=60)  # blocks until string is found, or 60s timeout
            if server_ready.is_set():
                print("------- Llama server is ready ---------")
            else:
                print("------- Timed out waiting for Llama server -------")
        else:
            print('Server is already running')

    def kill_llama_server(self):
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(self.server_process.pid)],
            check=False)

        self.server_process = None

    def ask(self, question):
        self.messages.append({"role": "user", "content": question})

        if self.grammar is not None:
            response = requests.post(self.URL, json={
                "messages": self.messages,
                "grammar": self.grammar,
                "temperature": 0.7
            })
        else:
            response = requests.post(self.URL, json={
                "messages": self.messages,
                "temperature": 0.7
            })

        reply = response.json()["choices"][0]["message"]["content"]
        self.messages.append({"role": "assistant", "content": reply})
        return reply


    def check_solution_code(self, test_input_canvas: Canvas, test_output_canvas: Canvas, index_of_reply: int = -1) -> tuple[float, List]:
        replies = [m['content'] for m in self.messages if m['role']=='assistant']
        reply = replies[index_of_reply]
        code = reply.split('```')[1].split('python')[1]
        exec(code, globals())
        predicted_canvas = solver(test_input_canvas)
        ratio_correct = np.where((predicted_canvas.actual_pixels  - test_output_canvas.actual_pixels) == 0)[0].size / test_output_canvas.actual_pixels.size
        return ratio_correct, predicted_canvas

    @staticmethod
    def get_docstrings(module: ModuleType, function_names: list[str] | None = None) -> str:
        if function_names is None:
            fns = [
                name for name, obj in inspect.getmembers(module, callable)
                if inspect.isfunction(obj) or inspect.isbuiltin(obj)
            ]
        else:
            if '.' in function_names[0]:
                fns = [fn.split('.')[-1] for fn in function_names]
            else:
                fns = function_names

        docstring = ''
        for name in fns:
            if hasattr(module, name) and callable(getattr(module, name)):
                docstring += f'{name}: {inspect.getdoc(getattr(module, name))}\n'

        docstring += '\n'
        return docstring

    def generate_prompt_for_task_and_dsl_funcs(self, task: Task, dsl_module: ModuleType, used_dsl_funcs: List[str] | None = None) -> str:
        prompt = (
            f'What follows is a series of arrays each representing an image with a low number of pixels. Each image\n'
            f'can range from 3x3 to 32x32 pixels. Each pixel can be one of ten possible colours.\n'
            f'Each pixel matters. There are {len(task.input_canvases)} training input - output pairs. There are two images for each training pair\n'
            f'denoted as Train Input N and Train Output N (where N is the number of the training pair).\n'
            f'All of the train pairs showcase a specific logic that if found and applied it will transform the input\n'
            f'image to the output one. I want you to generate the program that can do this transformation.\n'
            f'You have also been given a grammar to follow. Do a little bit of thinking and explain your logic \n'
            f'and then create a program that uses only the classes and functions described below. The program should\n'
            f'be formated as follows:\n'
            f'```\n'
            f'python\n'
            f'def solver(in_canvas):\n'
            f'    code\n'
            f'```\n\n'
            f"Here is a description of the allowed API:\n"
            f'The classes allowed are:\n'
            f'Canvas: An NxM image with exactly the same format as the input and output images given to you.\n'
            f'A Canvas also holds inside it separately Objects (see below). As these Objects update the Canvas\n'
            f'pixels also update.\n'
            f'Object: An KxL (smaller than NxM) image of pixels with a set of properties:\n'
            f'i) canvas_position: Where in a Canvas the bottom left pixel of the Object should be placed.\n'
            f'ii) colour: If the Objects pixels are all of the same colour this property is the int of that colour.\n'
            f'Distance2D: A vector that has magnitude (in pixels), direction (up, down, left , right, up-left, up-right\n'
            f'down-left, down-right) and an position (where the 0,0 of the vector is)\n'
            f'Point: A two integer tuple denoting the coordinates of a pixel on a Canvas. The first number is the x (horizontal)\n'
            f'coordinate and the second the y (vertical).\n'
            f'int: An integer.\n\n'
            f'The functions of the grammar are:\n')

        prompt += ModelClient.get_docstrings(module=dsl_module, function_names=used_dsl_funcs)

        prompt += (f'The colour mapping is: \n{Colour.Black} = Black, {Colour.Blue} = Blue, {Colour.Red} = Red, \n'
                   f'{Colour.Green} = Green, {Colour.Yellow} = Yellow, {Colour.Gray} = Gray, {Colour.Purple} = Purple,\n'
                   f'{Colour.Orange} = Orange, {Colour.Azure} = Azure, {Colour.Burgundy} = Burgundy.\n\n')

        prompt += task.get_task_arrays_for_llm_prompt()

        return prompt