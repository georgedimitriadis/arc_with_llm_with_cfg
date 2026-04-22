
import load_data
from visualization.visualize_data import plot_task, plot_pixels

task_name = '05f2a901'
data = load_data.load_task_data_as_pixels('05f2a901', with_white_background=False)
#plot_task(task_name)

example0 = data['train'][0]
input0 = example0['input']
output0 = example0['output']

def array_to_text(array):
    result = ""
    for ar in array:
        for i in ar:
            result += f'{i} '
        result += '\n'
    return result
