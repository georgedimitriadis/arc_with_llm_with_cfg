import numpy as np
import subprocess
import threading
import atexit
import requests
from subprocess import Popen

from structure.canvas.canvas import Canvas
from dsls.our_dsl.functions import dsl_functions as dsl

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


    def check_solution_code(self, test_input_canvas: Canvas, test_output_canvas: Canvas, index_of_reply: int = -1) -> float:
        replies = [m['content'] for m in self.messages if m['role']=='assistant']
        reply = replies[index_of_reply]
        code = reply.split('```')[1].split('python')[1]
        exec(code, globals())
        predicted_canvas = solver(test_input_canvas)
        ratio_correct = np.where((predicted_canvas.actual_pixels  - test_output_canvas.actual_pixels) == 0)[0].size / test_output_canvas.actual_pixels.size
        return ratio_correct, predicted_canvas
