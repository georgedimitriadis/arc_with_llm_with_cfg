
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

client = ModelClient()
prompt = client.generate_prompt_for_task_and_dsl_funcs(task, dsl, used_functions)

client.initialise_llama_server()
reply = client.ask(question=prompt)

ratio, out_canvas = client.check_solution_code(task.test_input_canvases[0], task.test_output_canvases[0])

client.kill_llama_server()




