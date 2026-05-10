
from types import ModuleType

from model_interaction.model_interaction import ModelClient
from structure.geometry.basic_geometry import Colour
from structure.task.task import Task

coding_assistant_context = (f"You are a Python coding assistant operating inside a sandboxed directory.\n\n"
                            f"AVAILABLE MODULE:\n"
                            f"A file called dsl_functions.py is already present in the sandbox. "
                            f"It contains all the DSL functions you are allowed to use."
                            f"Always import from it at the top of any code you write, like so:\n"
                            f"import dsl_functions as dsl\n\n"
                            f"YOUR WORKFLOW — always follow these steps:\n"
                            f"1. Use write_file to save your thinking and solution code as a .py file in the sandbox.\n"
                            f"2. Use run_python to execute it and observe the output.\n"
                            f"3. If there are errors or wrong results, fix the file and run again.\n"
                            f"4. Only report back to the user once the code runs correctly.\n\n"
                            f"RULES:\n"
                            f"- Never just write code as text — always use write_file then run_python.\n"
                            f"- Only use functions from dsl_functions.py. Do not import anything else. Not even numpy. The dsl_functions is all you need. Read them, understand them and use them.\n"
                            f"- All files must stay inside the sandbox.\n,"
                            f"- Do not run code that you have not previously saved in a script file. Also at every iteration create a new script. There needs to be a history of all your efforts.\n")

coding_assistant_context_no_dsl = (f"You are a Python coding assistant operating inside a sandboxed directory.\n\n"
                                   f"YOUR WORKFLOW — always follow these steps:\n"
                                    f"1. Use write_file to save your thinking and solution code as a .py file in the sandbox.\n"
                                    f"2. Use run_python to execute it and observe the output.\n"
                                    f"3. If there are errors or wrong results, fix the file and run again.\n"
                                    f"4. Only report back to the user once the code runs correctly.\n\n"
                                    f"RULES:\n"
                                    f"- Never just write code as text — always use write_file then run_python.\n"
                                    f"- All files must stay inside the sandbox.\n"
                                    f"- Do not run code that you have not previously saved in a script file. Also at every "
                                   f"iteration create a new script. There needs to be a history of all your efforts.\n")

def description_of_problem_prompt(task: Task):
    prompt = (f'What follows is a series of arrays each representing an image with a low number of pixels. Each image\n'
    f'can range from 3x3 to 32x32 pixels. Each pixel can be one of ten possible colours.\n'
    f'Each pixel matters. There are {len(task.input_canvases)} training input - output pairs. There are two images for each training pair\n'
    f'denoted as Train Input N and Train Output N (where N is the number of the training pair).\n'
    f'All of the train pairs showcase a specific logic that if found and applied it will transform the input\n'
    f'image to the output one. I want you to generate the program that can do this transformation.\n\n')

    return prompt

def colour_mapping_prompt():
    return (f'The colour mapping is: \n{Colour.Black} = Black, {Colour.Blue} = Blue, {Colour.Red} = Red, \n'
               f'{Colour.Green} = Green, {Colour.Yellow} = Yellow, {Colour.Gray} = Gray, {Colour.Purple} = Purple,\n'
               f'{Colour.Orange} = Orange, {Colour.Azure} = Azure, {Colour.Burgundy} = Burgundy.\n\n')

def generate_prompt_for_task_and_dsl_funcs_no_grammar(task: Task, dsl_module: ModuleType,
                                           used_dsl_funcs: list[str] | None = None) -> str:

    prompt = description_of_problem_prompt(task)
    prompt += (
        f'Create a function called solver that out of all the allowed functions in the dsl_functions.py script it uses \n'
        f'the functions described below. In the below description you will also find the docstrings of the functions.\n'
        f'For the solver function you are only allowed to use the allowed functions, and python if, for and while loops \n'
        f'and nothing else. The only other python keywords in the solver function should be the def and the return keywords.\n'
        f'You are allowed to create new variables but they should always equal a function. All functions\n'
        f'return something. Treat this as a functional paradigm code. \n'
        f'The solver function should be surrounded with #---Begin Solver---\\n and with #---End Solver---\\n comments.\n'
        f'After the solver function you can write any other code to help you understand if the solver is working or not.\n'
        f'The whole script should be formated as follows:\n\n'
        f'# Here goes all the required imports\n'
        f'#---Begin Solver---\n'
        f'def solver(in_canvas):\n'
        f'  # Here goes the code for the solver function\n'
        f'  return out_canvas\n'
        f'#---End Solver---\n\n'
        f'# Here goes any other code to test the solver function\n'
        f'\n\n'
        f'As mentioned in the RULES you are not to print out any of that code. You are to create a script in the sandbox,\n'
        f'write the code there using the write_file tool, use the run_python tool to run it, test if it is correct and\n'
        f'report the results. Stop once you have a solver that creates a correct out_canvas.\n'
        f"Here is a description of the allowed functions (and the classes they operate on):\n"
        f'The classes used are:\n'
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
        f'Out of the functions in the dsl_functions.py the ones allowed are:\n')

    prompt += ModelClient.get_docstrings(module=dsl_module, function_names=used_dsl_funcs)

    prompt += colour_mapping_prompt()

    prompt += (f'Always start with the generate_contiguous_objects function. This will create the correct objects\n'
               f'for this problem. Then use the other given functions to derive information from these objects\n'
               f'and subsequently manipulate these objects to their final positions and colour changes.\n'
               f'The objects created are all you need to solve the problem. You need to look into their individual\n'
               f'pixels to get information but manipulation is to be done only at the level of objects with the\n'
               f'provided functions. Do not try to manipulate individual pixels. Think only around objects and\n'
               f'how you should change them in order to solve the problem.\n\n')
    prompt += task.get_task_arrays_for_llm_prompt()

    return prompt



def generate_prompt_for_task_and_dsl_funcs_with_grammar(task: Task, dsl_module: ModuleType,
                                           used_dsl_funcs: list[str] | None = None) -> str:
    prompt = description_of_problem_prompt(task)

    prompt += (f'Create a function called solver that out of all the allowed functions in the dsl_functions.py script it uses \n'
        f'the functions described below. In the below description you will also find the docstrings of the functions.\n'
        f'For the solver function you are only allowed to use the allowed functions, and python if, for and while loops \n'
        f'and nothing else. The only other python keywords in the solver function should be the def and the return keywords.\n'
        f'You are allowed to create new variables but they should always equal a function. All functions\n'
        f'return something. Treat this as a functional paradigm code. \n'
        f'All of the above are described in the grammar given to you that constrains you to follow these rules. Use \n'
        f'descriptive names for the variables and do not use single letter names.\n'
        f'You can use python comments anywhere in the code to allow you to think about the problem\n.'
        f'Read carefully the dsl functions given to you and think how you can use them to solve the problem.\n'
        f'The whole script should be formated as follows:\n\n'
        f'# Here goes all the required imports\n'
        f'def solver(in_canvas):\n'
        f'  # Here goes the code for the solver function\n'
        f'  return out_canvas\n'
        f'\n\n'
        f'As mentioned in the RULES you are not to print out any of that code. You are to create a script in the sandbox,\n'
        f'write the code there using the write_file tool, use the run_python tool to run it, test if it is correct and\n'
        f'report the results. Stop once you have a solver that creates a correct out_canvas.\n'
        f"Here is a description of the allowed functions (and the classes they operate on):\n"
        f'The classes used are:\n'
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
        f'Out of the functions in the dsl_functions.py the ones allowed are:\n')

    prompt += ModelClient.get_docstrings(module=dsl_module, function_names=used_dsl_funcs)

    prompt += (f'The colour mapping is: \n{Colour.Black} = Black, {Colour.Blue} = Blue, {Colour.Red} = Red, \n'
               f'{Colour.Green} = Green, {Colour.Yellow} = Yellow, {Colour.Gray} = Gray, {Colour.Purple} = Purple,\n'
               f'{Colour.Orange} = Orange, {Colour.Azure} = Azure, {Colour.Burgundy} = Burgundy.\n\n')

    prompt += task.get_task_arrays_for_llm_prompt()

    return prompt


def generate_prompt_for_thinking_in_objects(task: Task):
    prompt = (
        f'I am going to give you {task.number_of_canvasses - 2} images, half being input images and half output. '
        f'Each image has a number of pixels (it can be from 3x3 to 32x32).'
        f' There is a logic that transforms the input to the output.'
        f' Each pixel in the images has one out of ten colours. I will give you the colour code between the numbers in the arrays'
        f' I will provide and the colours.\n'
    )

    prompt += colour_mapping_prompt()
    prompt += task.get_task_arrays_for_llm_prompt()

    prompt += (
        f'The pixels in the {task.number_of_canvasses - 2} images group together to form some objects. Each object is a '
        f'set of coloured pixels (not black). The objects can all be the same colour or can be of different colours. '
        f'Each object has a rectangular bounding box (that can include some of the black pixels which for the object '
        f'count as transparent) it has a canvas_pos (canvas position) as the bottom left pixel coordinate of the bounding'
        f' box in the image (also known as canvas).' 
        f'The object might or might not change from input to output. Some objects might stay the same, some might change, '
        f'some might vanish and some might not exist in the input and appear in the output. The changes an object can have are:\n'
        f'1) Translate a number of pixels to a new position.\n'
        f'2) Rotate 90, 180 or 270 degrees (around its canvas_pos).\n'
        f'3) Change colour of all or some of its pixels to one or more new colours.\n'
        f'4) Increase or decrease in overall size.\n\n'
        f'I want you to do two things in the order given:\n'
        f'FIRST: FIND ALL OBJECTS IN THE TWO IMAGES.'
        f'I want you to find the objects that are present in the input and output images. Find the canvas_pos and the '
        f'size of the bounding box (in pixels) of each one of them. To decide on the objects look at the pair of input - output'
        f' images and try to create objects that are both as large and as invariant between input-output as possible. Also try to create'
        f' as few objects as possible. The order of which changes preserve more invariance is (from more to less invariant):\n'
        f'1) An object that does not translate, rotate or change colour and exists in both images.\n'
        f'2) An object that either translates or rotates.\n'
        f'3) An object that changes colour.\n'
        f'4) An object that changes more than one colours.\n'
        f'5) An object that does more than one of the above changes (e.g. translates and changes colour).\n'
        f'6) An object that changes size.\n'
        f'7) An object that both changes size and does some of the other changes mentioned above.\n'
        f'8) An object that exists on one but not on both of the images.\n'
        f'DO NOT PROCEED UNTIL YOU HAVE IDENTIFIED THE OBJECTS IN THE IMAGES. AT EVERY ITERATION OF YOUR THINKING YOU MUST '
        f'FIRST FIND THE OBJECTS (OR USE THE ONES YOU HAVE FOUND FROM PREVIOUS ITERATIONS.\n\n'
        f'TWO: FIND THE LOGIC THAT TRANSFORMS THE INPUT OBJECTS TO GENERATE THE OUTPUT OBJECTS.\n'
        f'Once you have defined the objects in the above way notice that there is a logic that you can apply to the objects '
        f'of the input to create the output. This logic involves using cues from the input in order to define the transformations '
        f'of the objects from the input to the output. Find these cues and the transformations and write python code '
        f'that takes any of the input images and correctly creates the output. THE CODE MUST OPERATE ON THE OBJECTS YOU HAVE'
        f'FOUND AND TRANSFORM THOSE. The cues can come from anything in the input image. Some examples of where cues may lie:\n'
        f'1) Objects attributes\n'
        f'2) Relations between objects\n'
        f'3) Individual pixel attributes\n'
        f'4) Relations between pixels\n'
        f'5) Relations between objects and individual pixels\n'
        f'This is not exhaustive. Explore all of the above points if you cannot find a solution. Do not forget to check\n'
        f'not the properties of individual objects and pixels but also relationships between objects, between pixels and'
        f'between objects and pixels.\n\n'
        f'As the RULES say you are not to print out any of that code. You are to create scripts in the sandbox,\n'
        f'write the code there using the write_file tool, use the run_python tool to run it and test if it is correct\n'
        f'Stop once you have code that creates correctly all outputs.\n'
    )

    return prompt



def generate_prompt_for_thinking_in_objects_sequential(task: Task):
    prompts = []
    prompts.append(
        (
        f'I am going to give you images, one being input image and the other output. '
        f'Each image has a number of pixels (it can be from 3x3 to 32x32).'
        f' There is a logic that transforms the input to the output.'
        f' Each pixel in the images has one out of ten colours. I will give you the colour code between the numbers in the arrays'
        f' I will provide and the colours.\n'
        ) + colour_mapping_prompt() \
          + task.get_task_arrays_for_llm_prompt([0]) \
          +  (
        f'The pixels in the input image group together to form some objects. Each object is a '
        f'set of coloured pixels (not black). The objects can all be the same colour or can be of different colours. '
        f'Each object has a rectangular bounding box (that can include some of the black pixels which for the object '
        f'count as transparent) it has a canvas_pos (canvas position) as the bottom left pixel coordinate of the bounding'
        f' box in the image (also known as canvas). All colours except black count in the objects pixels. There are no coloured '
        f'pixels (non-black) that can be assumed to be transparent.\n'
        f'Objects may overlap each other (in this case in the input you will see only part of objects that are under other objects).\n'
        f'Each object can have a set of transformations applied to it. From this set there is a subset for each object '
        f'that when applied (to each object for all objects) then the resulting image matches the output image.'  
        f'The objects might or might not change from input to output. Some objects might stay the same, some might change.\n'
        f'The changes an object can have are:\n'
        f'1) Translate by a number of pixels to a new position.\n'
        f'2) Rotate 90, 180 or 270 degrees (around its canvas_pos).\n'
        f'3) Change colour of all or some of its pixels to one or more new colours.\n'
        f'4) Increase (by turning some black pixels outside the bounding box coloured and increasing the bounding box) '
        f'or decrease (by turning some coloured pixels black and decreasing the bounding box) in overall size.\n'
        f'Sometimes some transformations may take the part of the object outside the output canvas thus making it partially'
        f'visible.\n\n'
        f'FIND ALL OBJECTS IN THE INPUT IMAGE AND THEN FIND THE TRANSFORMATIONS YOU NEED TO APPLY TO GET THE RESULT LOOK '
        f'LIKE THE OUTPUT IMAGE.'
              )
                   )

    prompts.append(
        f'Using the exact same logic for the first input output pair find the objects and their transformations for the'
        f'following {3} pairs.\n\n' \
        + task.get_task_arrays_for_llm_prompt([1,2])
        )

    prompts.append(
        f'Read the code in the structure directory of your sandbox. In the object folder there is a script called object. '
        f'In there an Object class is defined. Create in a script objects of class Object that represent the objects you '
        f'have discovered for all {3} pairs in the input images for all examples.\n'
        f'Make sure you also fill in the Object.transformations with a list of ObjectTransformations encoding the '
        f'transformations you have discovered for each object.'
    )

    return prompts