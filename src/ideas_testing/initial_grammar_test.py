
import load_data
from ideas_testing import prompts
from structure.task.task import Task
from model_interaction.model_interaction import ModelClient

from dsls.our_dsl.functions import dsl_functions as dsl

from handwritten_solutions.function_recovery import get_called_functions
from handwritten_solutions import logic_functions_with_dsl as logic_funcs

task_name = 'a6f40cea' # '234bbc79', 'a6f40cea'
#logic_function = getattr(logic_funcs, f'logic_function_{task_name}')
#used_functions, docstrings = get_called_functions(logic_function)


data = load_data.load_task_data_as_pixels(task_name, with_white_background=False)
task = Task(data)

llm_sandbox = r'G:\Code\Repos\Mine\Machine_Learning\ARC\arc_with_llm_with_cfg\src\llm_sandbox'

coding_assistant_context = prompts.coding_assistant_context_no_dsl
#grammar = r'G:\Code\Repos\Mine\Machine_Learning\ARC\arc_with_llm_with_cfg\src\ideas_testing\grammar_normal_python_dsl_only.gbnf'
client = ModelClient(system_content=coding_assistant_context, sandbox_dir=llm_sandbox)
prompts = prompts.generate_prompt_for_thinking_in_objects_sequential(task)

client.initialise_llama_server()

#reply = client.ask(question=prompt)
prompt = prompts[0]
reply = client.ask_with_tools(question=prompt, verbose=True)

prompt =  prompts[1]
reply = client.ask_with_tools(question=prompt, verbose=True)

prompt =  prompts[2]
reply = client.ask_with_tools(question=prompt, verbose=True)

#ratio, out_canvas = client.check_solution_code(task.test_input_canvases[0], task.test_output_canvases[0])

client.kill_llama_server()




