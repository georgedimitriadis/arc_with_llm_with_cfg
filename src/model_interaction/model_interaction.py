import inspect
import json
from pathlib import Path
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
    TOOLS = [
        {
            "type": "function",
            "function": {
                "name": "run_python",
                "description": (
                    "Execute Python code and return stdout/stderr. "
                    "Use this to test code you write. "
                    "The working directory is the sandbox."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "Python code to execute."
                        }
                    },
                    "required": ["code"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "write_file",
                "description": "Write text content to a file inside the sandbox directory.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filename": {
                            "type": "string",
                            "description": "Filename relative to the sandbox (e.g. 'utils.py')."
                        },
                        "content": {
                            "type": "string",
                            "description": "Text content to write."
                        }
                    },
                    "required": ["filename", "content"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file from inside the sandbox directory.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filename": {
                            "type": "string",
                            "description": "Filename relative to the sandbox (e.g. 'utils.py')."
                        }
                    },
                    "required": ["filename"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "list_files",
                "description": "List all files currently in the sandbox directory.",
                "parameters": {
                    "type": "object",
                    "properties": {}
                }
            }
        }
    ]

    def __init__(self, system_content:str = "You are a helpful coding assistant.",
                 grammar_file: str | None = None,
                 sandbox_dir: str | Path = "./sandbox",
                 code_timeout: int = 5,
                 temperature = 0.7):

        self.URL = "http://127.0.0.1:8080/v1/chat/completions"
        if grammar_file is not None:
            self.grammar = open(grammar_file).read()
        else:
            self.grammar = None

        self.sandbox_dir = Path(sandbox_dir).resolve()
        self.sandbox_dir.mkdir(parents=True, exist_ok=True)
        self.code_timeout = code_timeout
        self.temperature = temperature

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


    # ── Sandbox Utilities ──────────────────────────────────────────────────────

    def _safe_path(self, filename: str) -> Path:
        """Resolve path and reject anything that escapes the sandbox."""
        target = (self.sandbox_dir / filename).resolve()
        if not str(target).startswith(str(self.sandbox_dir)):
            raise PermissionError(f"Path '{filename}' escapes the sandbox.")
        return target

    # ── Tool Implementations ───────────────────────────────────────────────────

    def _tool_run_python(self, code: str) -> str:
        try:
            result = subprocess.run(
                ["python", "-c", code],
                capture_output=True,
                text=True,
                timeout=self.code_timeout,
                cwd=self.sandbox_dir,
            )
            out = result.stdout
            if result.stderr:
                out += "\n[stderr]\n" + result.stderr
            return out.strip() or "(no output)"
        except subprocess.TimeoutExpired:
            return f"[error] Execution timed out after {self.code_timeout}s."
        except Exception as e:
            return f"[error] {e}"

    def _tool_write_file(self, filename: str, content: str) -> str:
        try:
            path = self._safe_path(filename)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            return f"Written {len(content)} chars to '{filename}'."
        except PermissionError as e:
            return f"[error] {e}"
        except Exception as e:
            return f"[error] {e}"

    def _tool_read_file(self, filename: str) -> str:
        try:
            path = self._safe_path(filename)
            if not path.exists():
                return f"[error] File '{filename}' does not exist."
            return path.read_text(encoding="utf-8")
        except PermissionError as e:
            return f"[error] {e}"
        except Exception as e:
            return f"[error] {e}"

    def _tool_list_files(self) -> str:
        files = sorted(self.sandbox_dir.rglob("*"))
        if not files:
            return "(sandbox is empty)"
        return "\n".join(
            str(f.relative_to(self.sandbox_dir)) + ("/" if f.is_dir() else "")
            for f in files
        )

    def _dispatch_tool(self, name: str, args: dict) -> str:
        if name == "run_python":
            return self._tool_run_python(args["code"])
        elif name == "write_file":
            return self._tool_write_file(args["filename"], args["content"])
        elif name == "read_file":
            return self._tool_read_file(args["filename"])
        elif name == "list_files":
            return self._tool_list_files()
        else:
            return f"[error] Unknown tool: {name}"

    # ── API Calls ──────────────────────────────────────────────────────────────

    def _post(self, extra: dict | None = None) -> dict:
        """Base POST to the model. Merges any extra fields into the payload."""
        payload = {"messages": self.messages, "temperature": self.temperature}

        if self.grammar is not None:
            payload["grammar"] = self.grammar

        if extra:
            payload.update(extra)

        response = requests.post(self.URL, json=payload, timeout=1200)
        response.raise_for_status()
        return response.json()["choices"][0]


    def ask(self, question):
        """Single-turn ask with no tool use."""
        self.messages.append({"role": "user", "content": question})
        choice = self._post()
        reply = choice["message"]["content"]
        self.messages.append({"role": "assistant", "content": reply})
        return reply

    def ask_with_tools(self, question: str, verbose: bool = True) -> str:
        """
        Agentic ask: the model can call tools autonomously and will loop
        until it emits a plain text response with no tool calls.
        Returns the final text response.
        """
        self.messages.append({"role": "user", "content": question})

        final_reply = ""
        while True:
            choice = self._post(extra={"tools": self.TOOLS, "tool_choice": "auto"})
            msg = choice["message"]
            tool_calls = msg.get("tool_calls") or []

            # Always append the assistant turn to history
            self.messages.append(msg)

            if msg.get("content") and verbose:
                print(f"\nQwen: {msg['content']}")

            if not tool_calls:
                # Model is done — no more tool calls
                final_reply = msg.get("content", "")
                break

            # Execute each tool call and feed results back
            for tc in tool_calls:
                fn_name = tc["function"]["name"]
                fn_args = json.loads(tc["function"]["arguments"])

                if verbose:
                    arg_preview = ", ".join(
                        f"{k}={repr(v)[:1000]}" for k, v in fn_args.items()
                    )
                    print(f"\n  ▶ tool: {fn_name}({arg_preview})")

                result = self._dispatch_tool(fn_name, fn_args)

                if verbose:
                    preview = result[:1000] + ("..." if len(result) > 1000 else "")
                    print(f"  ◀ result: {preview}")

                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result,
                })

        return final_reply

    def reset_conversation(self):
        """Clear message history, keeping only the system prompt."""
        self.messages = self.messages[:1]


    def check_solution_code(self, test_input_canvas: Canvas, test_output_canvas: Canvas, index_of_reply: int = -1) -> tuple[float, List]:
        replies = [m['content'] for m in self.messages if m['role']=='assistant']
        reply = replies[index_of_reply]
        code = reply.split('```')[1].split('python')[1]
        exec(code, globals())
        predicted_canvas = solver(test_input_canvas)
        ratio_correct = np.where((predicted_canvas.actual_pixels  - test_output_canvas.actual_pixels) == 0)[0].size / test_output_canvas.actual_pixels.size
        return ratio_correct, predicted_canvas


    # ── Static Helpers (unchanged) ─────────────────────────────────────────────

    @staticmethod
    def get_docstrings(module: ModuleType, function_names: list[str] | None = None) -> str:
        if function_names is None:
            fns = [
                name for name, obj in inspect.getmembers(module, callable)
                if inspect.isfunction(obj) or inspect.isbuiltin(obj)
            ]
        else:
            fns = [fn.split(".")[-1] if "." in fn else fn for fn in function_names]

        docstring = ""
        for name, ext_name in zip(fns, function_names):
            if hasattr(module, name) and callable(getattr(module, name)):
                docstring += f"{ext_name}: {inspect.getdoc(getattr(module, name))}\n"

        return docstring + "\n"

    def generate_prompt_for_task_and_dsl_funcs(self, task: Task, dsl_module: ModuleType, used_dsl_funcs: List[str] | None = None) -> str:
        grammar = ''
        if self.grammar:
            grammar = 'You have also been given a grammar to follow. '
        prompt = (
            f'What follows is a series of arrays each representing an image with a low number of pixels. Each image\n'
            f'can range from 3x3 to 32x32 pixels. Each pixel can be one of ten possible colours.\n'
            f'Each pixel matters. There are {len(task.input_canvases)} training input - output pairs. There are two images for each training pair\n'
            f'denoted as Train Input N and Train Output N (where N is the number of the training pair).\n'
            f'All of the train pairs showcase a specific logic that if found and applied it will transform the input\n'
            f'image to the output one. I want you to generate the program that can do this transformation.\n'
            f'{grammar}Do a little bit of thinking and explain your logic \n'
            f'and then create a function called solver that uses only the classes and functions described in the API below.\n'
            f'For the solver function you are only allowed to use the API functions, and python if, for and while loops \n'
            f'and nothing else. The only other python keywords in the solver function should be the def and the return keywords.\n'
            f'You are allowed to create new variables but they should always equal an API function. All API functions\n'
            f'return something. Treat this as a functional paradigm code. \n'
            f'The solver function should be surrounded with #---Begin Solver---\n'
            f'and with #---End Solver---\n comments. After the solver function you can write any other code to help you\n'
            f'understand if the solver is working or not.\n'
            f'The whole script should be formated as follows:\n'
            f'```\n'
            f'python\n'
            f'# Here goes all the required imports\n'
            f'from dsls.our_dsl.functions import dsl_functions as dsl\n\n'
            f'#---Begin Solver---\n'
            f'def solver(in_canvas):\n'
            f'  # Here goes the code for the solver function\n'
            f'  return out_canvas\n'
            f'#---End Solver---\n\n'
            f'# Here goes any other code to test the solver function\n'
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