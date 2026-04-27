
import load_data
from structure.task.task import Task
from model_interaction.model_interaction import ModelClient

task_name = '05f2a901'
data = load_data.load_task_data_as_pixels(task_name, with_white_background=False)

task = Task(data)
prompt = task.generate_prompt_for_llm()

client = ModelClient()
#grammar_file = r'G:\Code\Repos\Mine\Machine_Learning\ARC\arc_with_llm_with_cfg\src\ideas_testing\grammar.gbnf'
#client = ModelClient(grammar_file=grammar_file)

client.initialise_llama_server()

reply = client.ask(question=prompt)

ratio, out_canvas = client.check_solution_code(task.test_input_canvases[0], task.test_output_canvases[0])

client.kill_llama_server()



