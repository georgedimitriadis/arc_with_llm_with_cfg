
"""
The module assumes that the project/data directory is in the PYTHONPATH (made a source folder in PyCharm).
"""

import json
import numpy as np
import os

data_folder = os.path.dirname(os.path.realpath(__file__))

def find_arc_version(task_name:str) -> int | None:
    """
    Finds in which ARC version (1 or 2) the task exists. If it exists in both it returns 1.
    :param task_name: The name of the task.
    :return: 1 or 2
    """
    for i in [1, 2]:
        if os.path.isfile(os.path.join(data_folder, f'arc_{i}/evaluation/{task_name}.json')) or \
                os.path.isfile(os.path.join(data_folder, f'arc_{i}/training/{task_name}.json')):
            return i

def find_eval_or_train(task_name:str) -> str | None:
    """
    Find if the task exists in the training or the evaluation folders of an ARC dataset. If it exists in both it returns training.
    :param task_name: The name of the task.
    :return: 'training' or 'evaluation'
    """
    for i in [1, 2]:
        for place in ['training', 'evaluation']:
            if os.path.isfile(os.path.join(data_folder, f'arc_{i}/{place}/{task_name}.json')):
                return place

def load_task_data_as_pixels(task_name:str, with_white_background:bool = True,
                             version:int = None, place: str = None):
    """
    This loads the data in the following form. A dict of {'train':[{'input':NDArray, 'output':NDArray}, ...],
    'test': [{'input':NDArray, 'output':NDArray}, ...]}. Each NDArray is a NxM array of integers that represent the
    basic_geometry.Colour colours.
    :param task_name: The name of the task.
    :param version: The version of the ARC competition. If None the function will search for the task first in ARC 1 and then in ARC 2.
    :param place: The training or evaluation folder of the competition data. If None the function will search for the task first in the training folder.
    :return: The dict of data as described above.
    """
    if version is None:
        version = find_arc_version(task_name)
    if place is None:
        place = find_eval_or_train(task_name)

    filename = os.path.join(data_folder, f'arc_{version}/{place}/{task_name}.json')
    assert os.path.isfile(filename), print(f'{filename} does not exist')

    with open(filename, 'r') as fp:
        data = json.load(fp)

    for tt in ['train', 'test']:
        for io in ['input', 'output']:
            for example in range(len(data[tt])):
                data_to_copy = np.array(data[tt][example][io]) + 1
                if with_white_background:
                    data[tt][example][io] = np.zeros((32, 32)).astype(int)
                    data[tt][example][io][:data_to_copy.shape[0], :data_to_copy.shape[1]] = data_to_copy
                else:
                    data[tt][example][io] = data_to_copy

    return data

