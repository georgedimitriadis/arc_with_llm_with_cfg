
import load_data
from visualization.visualize_data import plot_task

task_name = '05f2a901'
data = load_data.load_task_data_as_pixels('05f2a901')

plot_task(task_name)

