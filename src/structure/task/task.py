
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

    def generate_prompt_for_llm(self):
        prompt = (f'What follows is a series of arrays each representing an image with a low number of pixels. Each image\n'
                  f'can range from 3x3 to 32x32 pixels. Each pixel can be one of ten possible colours.\n'
                  f'Each pixel matters. There are {len(self.input_canvases)} training input - output pairs. There are two images for each training pair\n'
                  f'denoted as Train Input N and Train Output N (where N is the number of the training pair).\n'
                  f'All of the train pairs showcase a specific logic that if found and applied it will transform the input\n'
                  f'image to the output one. I want you to generate the program that can do this transformation.\n'
                  f'You have also been given a grammar to follow. Do a little bit of thinking and explain your logic \n'
                  f'and then create a program that uses only the classes and functions described below. The program should\n'
                  f'be formated as follows:\n'
                  f'```\n'
                  f'python\n'
                  f'def solver(in_canvas):\n'
                  f'    code\n'
                  f'```\n\n'
                  f"Here is a description of the allowed API:\n"
                  f'The classes allowed are:\n'
                  f'Canvas: An NxM image with exactly the same format as the input and output images given to you.\n'
                  f'A Canvas also holds inside it separately Objects (see below). As these Objects update the Canvas\n'
                  f'pixels also update.\n'
                  f'Object: An KxL (smaller than NxM) image of pixels with a set of properties:\n'
                  f'i) canvas_position: Where in a Canvas the bottom left pixel of the Object should be placed.\n'
                  f'ii) colour: If the Objects pixels are all of the same colour this property is the int of that colour.\n'
                  f'Distance: A vector that has magnitude (in pixels), direction (up, down, left , right, up-left, up-right\n'
                  f'down-left, down-right) and an position (where the 0,0 of the vector is)\n'
                  f'int: An integer.\n'
                  f'The functions of the grammar are:\n'
                  f'dsl.make_new_canvas_as(Canvas) -> Canvas: It makes a new Canvas with all pixels Black (1) of the same\n'
                  f'size as the input Canvas.\n'
                  f'dsl.add_object_to_canvas(Canvas Object) -> Canvas: It adds the Object onto the Canvas.\n'
                  f'dsl.generate_contiguous_colour_objects(Canvas) -> Canvas: It takes a Canvas with some coloured pixels\n'
                  f'but no Objects inside it and breaks up the contiguous groups of pixels with the same colour into\n'
                  f'separate Objects that get added to the Canvas. This is the function that takes just pixels as given\n'
                  f'by the problem and generates the Objects that all other functions will operate on\n.'
                  f'dsl.select_only_object_of_colour(Canvas, int) -> Object: It returns the Object in the Canvas of colour int.\n'
                  f'dsl.object_transform_translate_along_direction(Object, Distance) -> Object: It translates the Object along the\n'
                  f'given Distance.\n'
                  f'dsl.get_distance_touching_between_objects(Object1, Object2) -> Distance: It finds the Distance that Object1\n'
                  f'needs to move to touch Object2.\n\n'
                  f'The colour mapping is: \n{Colour.Black} = Black, {Colour.Blue} = Blue, {Colour.Red} = Red, \n'
                  f'{Colour.Green} = Green, {Colour.Yellow} = Yellow, {Colour.Gray} = Gray, {Colour.Purple} = Purple,\n'
                  f'{Colour.Orange} = Orange, {Colour.Azure} = Azure, {Colour.Burgundy} = Burgundy.\n\n'
                  f'The images are:\n')
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


