
from __future__ import annotations

from typing import List
from structure.geometry.basic_geometry import Colour

from matplotlib import pyplot as plt
import numpy as np
from copy import copy

from structure.canvas.canvas import Canvas


class Task:

    def __init__(self, task_data: dict[str, list]):
        
        self.raw_data = task_data
        self.number_of_io_pairs = len(task_data['train']) + len(task_data['test'])
        self.number_of_canvasses = self.number_of_io_pairs * 2

        self.input_canvases = []
        self.output_canvases = []
        self.test_input_canvases = []
        self.test_output_canvases = []

        self.task_id = None
        self.task_description = ''

        self.create_canvases_from_data()


    def __copy__(self):
        new_task = Task(task_data=self.raw_data)

        for i, (ic, oc) in enumerate(zip(self.input_canvases, self.output_canvases)):
            new_task.input_canvases[i] = copy(ic)
            new_task.output_canvases[i] = copy(oc)
        for i, (tic, toc) in enumerate(zip(self.test_input_canvases, self.test_output_canvases)):
            new_task.test_input_canvases[i] = copy(tic)
            new_task.test_output_canvases[i] = copy(toc)
        new_task.task_id = copy(self.task_id)
        new_task.task_description = copy(self.task_description)

        return new_task


    def create_canvases_from_data(self):
        """
        This takes the raw data for every example and test canvas and creates a new Canvas adding the raw data to the
        new Canvas' actual_pixels. At this point there are no Objects defined within the Canvases.
        :return:
        """
        for t in self.raw_data:
            for i in range(len(self.raw_data[t])):
                for io in self.raw_data[t][i]:
                    data = Canvas(actual_pixels=self.raw_data[t][i][io])
                    if io == 'input' and t == 'train':
                        self.input_canvases.append(data)
                    elif io == 'output' and t == 'train':
                        self.output_canvases.append(data)
                    elif io == 'input' and t == 'test':
                        self.test_input_canvases.append(data)
                    elif io == 'output' and t == 'test':
                        self.test_output_canvases.append(data)


    def generate_contiguous_objects_by_colour(self):
        """
        Goes through all the Canvases of the Task and for each one creates a number of Objects, each defined as a contiguous
        blob of pixels of the same colour.
        :return:
        """
        colours = self.get_all_colours()
        colours = np.sort(colours)
        colour_to_id = {c:i for i, c in enumerate(colours)}
        for ic, oc in zip(self.input_canvases, self.output_canvases):
            ic.generate_contiguous_objects_by_colour(colour_to_id)
            oc.generate_contiguous_objects_by_colour(colour_to_id)
        for tic, toc in zip(self.test_input_canvases, self.test_output_canvases):
            tic.generate_contiguous_objects_by_colour(colour_to_id)
            toc.generate_contiguous_objects_by_colour(colour_to_id)


    def get_all_colours(self) -> List[int]:
        pass
        colours = set()
        for i, o in zip(self.input_canvases, self.output_canvases):
            colours.update(set(i.get_used_colours()))
            colours.update(set(o.get_used_colours()))
        for i in self.test_input_canvases:
            colours.update(set(i.get_used_colours()))

        return list(colours)

    def array_to_text(self, array):
        result = ""
        for ar in array:
            for i in ar:
                result += f'{int(i)} '
            result += '\n'
        return result


    def get_task_arrays_for_llm_prompt(self) -> str:

        prompt = f'The images are:\n'
        for i, (ic, oc) in enumerate(zip(self.input_canvases, self.output_canvases)):
            icp = ic.actual_pixels
            ocp = oc.actual_pixels
            prompt += f'Train Input {i}\n{self.array_to_text(icp)}\nTrain Output {i}\n{self.array_to_text(ocp)}\n\n'

        return prompt


    def show(self, canvas_index: int | str = 'all', save_as: str | None = None, two_cols: bool = False):
        """
        Shows some (canvas_index is int or 'test') or all (canvas_index = 'all') the Canvases of the Experiment
        :param two_cols: If False it will show the test data in a third column. Otherwise it will show the test data under the train
        :param save_as: If not None then save the figure generated as save_as file (but do not show it).
        :param canvas_index: Which Canvases to show (int, 'test' or 'all')
        :return:
        """

        thin_lines = True
        if save_as is None:
            thin_lines = False

        if type(canvas_index) == int:
            if canvas_index % 2 == 0:
                canvas = self.input_canvases[canvas_index // 2]
            else:
                canvas = self.output_canvases[canvas_index // 2]
            canvas.show(save_as=save_as)

        elif canvas_index == 'test':
            self.test_input_canvas.show(save_as=save_as)

        elif canvas_index == 'all':
            fig = plt.figure(figsize=(6, 16))
            index = 1
            nrows = self.number_of_io_pairs if two_cols else int(np.ceil(self.number_of_io_pairs / 2))
            ncoloumns = 2 if two_cols else 4

            for p in range(len(self.input_canvases)):
                self.input_canvases[p].show(fig_to_add=fig, nrows=nrows, ncoloumns=ncoloumns, index=index,
                                            thin_lines=thin_lines)
                self.output_canvases[p].show(fig_to_add=fig, nrows=nrows, ncoloumns=ncoloumns, index=index + 1,
                                             thin_lines=thin_lines)
                index += 2
            for p in range(len(self.test_input_canvases)):
                try:
                    self.test_input_canvases[p].show(fig_to_add=fig, nrows=nrows, ncoloumns=ncoloumns, index=index,
                                                thin_lines=thin_lines)
                except IndexError:
                    pass
                try:
                    if self.test_output_canvases[p] is not None:
                        self.test_output_canvases[p].show(fig_to_add=fig, nrows=nrows, ncoloumns=ncoloumns, index=index + 1,
                                                     thin_lines=thin_lines)
                except IndexError:
                    pass
                index += 2

            plt.tight_layout(pad=0.01)

            if save_as is not None:
                fig.savefig(save_as, dpi=5000)
                plt.close('all')


