
import load_data
from structure.task.task import Task
from model_interaction.model_interaction import ModelClient

from dsls.our_dsl.functions import dsl_functions as dsl

from handwritten_solutions.function_recovery import get_called_functions
from handwritten_solutions import logic_functions_with_dsl as logic_funcs

task_name = '234bbc79'
logic_function = getattr(logic_funcs, f'logic_function_{task_name}')
used_functions, docstrings = get_called_functions(logic_function)


data = load_data.load_task_data_as_pixels(task_name, with_white_background=False)
task = Task(data)


llm_sandbox = r'G:\Code\Repos\Mine\Machine_Learning\ARC\arc_with_llm_with_cfg\src\llm_sandbox'
coding_assistant_context = (f"You are a Python coding assistant operating inside a sandboxed directory.\n\n"
                            f"AVAILABLE MODULE:\n"
                            f"A file called dsl_functions.py is already present in the sandbox. "
                            f"It contains all the DSL functions you are allowed to use. "
                            f"Always import from it at the top of any code you write, like so:\n"
                            f"import dsl_functions as dsl\n\n"
                            f"YOUR WORKFLOW — always follow these steps:\n"
                            f"1. Use write_file to save your solution as a .py file in the sandbox.\n"
                            f"2. Use run_python to execute it and observe the output.\n"
                            f"3. If there are errors or wrong results, fix the file and run again.\n"
                            f"4. Only report back to the user once the code runs correctly.\n\n"
                            f"RULES:\n"
                            f"- Never just write code as text — always use write_file then run_python.\n"
                            f"- Only use functions from dsl_functions.py. Do not import anything else.\n"
                            f"- All files must stay inside the sandbox.\n")
client = ModelClient(system_content=coding_assistant_context, sandbox_dir=llm_sandbox)
prompt = client.generate_prompt_for_task_and_dsl_funcs(task, dsl, used_functions)

client.initialise_llama_server()

#reply = client.ask(question=prompt)
reply = client.ask_with_tools(question=prompt, verbose=True)

ratio, out_canvas = client.check_solution_code(task.test_input_canvases[0], task.test_output_canvases[0])

client.kill_llama_server()




