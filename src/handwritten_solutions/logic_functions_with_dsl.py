
from copy import copy

import numpy as np

from structure.canvas.canvas import Canvas
from structure.geometry.basic_geometry import Dimension2D, Point, Colour, Orientation, Surround, RelativePoint
from structure.object.primitives import Random
from dsls.our_dsl.functions import dsl_functions as dsl
from structure.task.task import Task


# <editor-fold desc="05f2a901">
def logic_function_05f2a901(in_canvas: Canvas) -> Canvas | None:
    try:
        in_canvas = dsl.generate_contiguous_colour_objects(in_canvas)
        out_canvas = dsl.make_new_canvas_as(in_canvas)

        moving_colour = 3
        steady_colour = 9

        moving = dsl.select_only_object_of_colour(in_canvas, colour=moving_colour)
        staying = dsl.select_only_object_of_colour(in_canvas, colour=steady_colour)
        dist = dsl.get_distance_touching_between_objects(moving, staying)

        updated_moving = dsl.object_transform_translate_along_direction(moving, dist)
        out_canvas = dsl.add_object_to_canvas(out_canvas, updated_moving)
        out_canvas = dsl.add_object_to_canvas(out_canvas, staying)

        return out_canvas
    except:
        return None

def single_line_05f2a901(in_canvas:Canvas) -> Canvas:
    return dsl.add_object_to_canvas(dsl.add_object_to_canvas(dsl.make_new_canvas_as(dsl.generate_contiguous_colour_objects(in_canvas)), dsl.object_transform_translate_along_direction(dsl.select_only_object_of_colour(dsl.generate_contiguous_colour_objects(in_canvas), 3),  dsl.get_distance_touching_between_objects(dsl.select_only_object_of_colour(dsl.generate_contiguous_colour_objects(in_canvas), 3), dsl.select_only_object_of_colour(dsl.generate_contiguous_colour_objects(in_canvas), 9)))), dsl.select_only_object_of_colour(dsl.generate_contiguous_colour_objects(in_canvas), 9))


def logic_function_05f2a901_llm_result(input_canvas):
    '''
    ### Logic Explanation
    By analyzing the three training pairs, a consistent spatial transformation emerges:
    1. **Object Identification**: Each input contains exactly two non-black contiguous shapes: an **Azure (9)** block
    and a **Red (3)** block. The Azure block acts as a stationary anchor, while the Red block is the moving agent.
    2. **Pixel-to-Object Conversion**: The raw pixel arrays must first be parsed into discrete `Object` entities.
    The function `generate_contiguous_colour_objects` handles this by grouping adjacent pixels of the same color.
    3. **Distance Calculation**: We need to find the exact translation vector that will make the Red object share an
    edge with the Azure object. `get_distance_touching_between_objects(mover, target)` computes this vector.
    4. **Transformation**: The computed distance is applied to the Red object using `object_transform_translate_along_direction`,
    producing a new translated `Object`.
    5. **Rendering**: Since translations return new objects rather than mutating existing ones, we initialize a fresh
    black canvas of the same dimensions using `make_new_canvas_as`. We then place both the translated Red object and the
    stationary Azure object onto this new canvas using `add_object_to_canvas`.
    This logic perfectly aligns with all three training examples:
    - **Train 0**: Red is below Azure → moves **UP** to touch.
    - **Train 1**: Red is left of Azure → moves **RIGHT** to touch.
    - **Train 2**: Red is above Azure → moves **DOWN** to touch.
    '''
    # 1. Convert contiguous pixel groups into discrete Objects
    canvas = dsl.generate_contiguous_colour_objects(input_canvas)

    # 2. Identify the stationary target (Azure/9) and the moving object (Red/3)
    target = dsl.select_only_object_of_colour(canvas, 9)
    mover = dsl.select_only_object_of_colour(canvas, 3)

    # 3. Calculate the exact translation vector for the mover to touch the target
    translation = dsl.get_distance_touching_between_objects(mover, target)

    # 4. Apply the translation to the mover, creating a new transformed Object
    moved_mover = dsl.object_transform_translate_along_direction(mover, translation)

    # 5. Initialize a fresh black canvas and render both objects onto it
    output_canvas = dsl.make_new_canvas_as(canvas)
    output_canvas = dsl.add_object_to_canvas(output_canvas, moved_mover)
    output_canvas = dsl.add_object_to_canvas(output_canvas, target)

    return output_canvas
# </editor-fold>


# <editor-fold desc="234bbc79">
def logic_function_234bbc79(in_canvas: Canvas) -> Canvas | None:
    try:
        out_canvas = dsl.make_new_canvas_as(in_canvas)
        temp_canvas = dsl.generate_contiguous_objects(in_canvas)
        objects = dsl.get_all_objects_in_canvas(temp_canvas)
        ordered_objects = dsl.order_objects_over_x(objects)
        list_length = dsl.length_of_list(ordered_objects)
        steps = dsl.subtract(list_length, 1)

        for i in dsl.arange(steps):
            o_before = dsl.get_item_n_from_list(ordered_objects, i)
            next_index = dsl.sum(i, 1)
            o_after = dsl.get_item_n_from_list(ordered_objects, next_index)

            before_grey_pixels = dsl.get_object_feature_position_of_colour(o_before, 6)
            before_grey_pixels = dsl.order_pixels_over_x(before_grey_pixels)
            before_grey_pixel = dsl.get_last_item(before_grey_pixels)

            after_grey_pixels = dsl.get_object_feature_position_of_colour(o_after, 6)
            after_grey_pixels = dsl.order_pixels_over_x(after_grey_pixels)
            after_grey_pixel = dsl.get_first_item(after_grey_pixels)

            canvas_pos = dsl.get_object_feature_canvas_pos(o_after)
            pos_pixel_diff = dsl.subtract_points(after_grey_pixel, canvas_pos)
            one_right_point = dsl.make_new_point(1, 0, 0)
            target_canvas_pos = dsl.sum_points(before_grey_pixel, one_right_point)
            new_canvas_pos = dsl.subtract_points(target_canvas_pos, pos_pixel_diff)

            o_after = dsl.object_transform_translate_to_point(o_after, new_canvas_pos)
            ordered_objects = dsl.replace_n_item_in_list(ordered_objects, next_index, o_after)

        for o in ordered_objects:
            colours = dsl.get_object_feature_all_colours(o)
            for colour in colours:
                if dsl.not_equal(colour, 6):
                    o = dsl.object_transform_new_colour(o, colour)
            out_canvas = dsl.add_object_to_canvas(out_canvas, o)

        last_obj_size = dsl.get_object_feature_size_x(o_after)
        last_obj_x = dsl.get_object_feature_canvas_pos_x(o_after)
        canvas_x_size = dsl.sum(last_obj_size, last_obj_x)
        canvas_y_size = dsl.get_canvas_feature_size_y(out_canvas)
        new_canvas_size = dsl.make_new_dimension2d(canvas_x_size, canvas_y_size)
        out_canvas = dsl.resize_canvas(out_canvas, new_canvas_size)

        return out_canvas
    except:
        return None
# </editor-fold>

# <editor-fold desc="Old Code">
def logic_function_007bbfb7(in_canvas: Canvas) -> Canvas:
    in_canvas_size_x = dsl.get_canvas_feature_size_x(in_canvas)
    in_canvas_size_y = dsl.get_canvas_feature_size_y(in_canvas)
    out_canvas_size_x = dsl.multiply(in_canvas_size_x, in_canvas_size_x)
    out_canvas_size_y = dsl.multiply(in_canvas_size_y, in_canvas_size_y)
    out_canvas_size = dsl.make_new_dimension2d(out_canvas_size_x, out_canvas_size_y)
    out_canvas = dsl.make_new_canvas(size=out_canvas_size)

    obj = dsl.select_largest_object_by_area(in_canvas)
    pos = dsl.get_object_feature_coloured_positions(obj)
    mult = dsl.get_canvas_feature_size(in_canvas)

    for p in pos:
        o_obj = dsl.copy_object(obj)
        new_point = dsl.mat_multiply_point(p, mult)
        o_obj = dsl.object_transform_translate_to_point(o_obj, new_point)
        out_canvas = dsl.add_object_to_canvas(out_canvas, o_obj)

    return out_canvas


#TODO: The fill_colour is given by looking at all the Example Outputs and figuring out the common colour for the fills.
#TODO: I do not know how to do this yet with this logic.
'''
def logic_function_00d62c1b(in_canvas: Canvas) -> Canvas:
    out_canvas = dsl.make_new_canvas_as(in_canvas)
    objects = dsl.select_all_objects(in_canvas)
    for obj in objects:
        obj = dsl.copy_object(obj)
        obj = dsl.object_transform_fill_holes(obj, fill_colour)
        out_canvas = dsl.add_object_to_canvas(out_canvas, obj)

    return out_canvas
'''


#  TODO: This logic is wrong. Correct it
def logic_function_017c7c7b(in_canvas: Canvas) -> Canvas:
    new_canvas_dimension = dsl.multiply_dimension_y(in_canvas.size, mult=1.5)
    out_canvas = dsl.make_new_canvas(size=new_canvas_dimension)
    obj = dsl.select_largest_object_by_area(in_canvas)
    obj_bottom, obj_top = dsl.object_transform_split_object_along_axis(obj=obj, cut_orientation=Orientation.Left,
                                                                       percentage=0.5)
    base_point = dsl.make_new_point(0, 0, 0)
    obj_top = dsl.object_transform_translate_to_point(obj_top, base_point)
    point_over = dsl.make_new_point(0, dsl.get_object_feature_size_y(obj_top), 0)
    obj = dsl.object_transform_translate_to_point(obj, point_over)
    out_canvas = dsl.add_object_to_canvas(out_canvas, obj)
    out_canvas = dsl.add_object_to_canvas(out_canvas, obj_top)

    top_point = dsl.make_new_point(0, dsl.divide(dsl.get_canvas_feature_size_y(in_canvas), 2), 0)
    obj = dsl.object_transform_translate_to_point(obj, top_point)
    out_canvas = dsl.add_object_to_canvas(out_canvas, obj)
    dsl.get_point_and_rotation_for_best_match_to_objects(obj, [obj], match_shape_only=True)
    return out_canvas



def logic_function_025d127b(in_canvas: Canvas) -> Canvas:
    out_canvas = dsl.make_new_canvas_as(in_canvas)
    for obj in dsl.select_all_objects(in_canvas):
        horizontal = dsl.make_new_orientation('Left')
        vertical = dsl.make_new_orientation('Up')

        obj_bottom, obj_base = dsl.object_transform_split_object_along_axis(obj=obj, cut_orientation=horizontal,
                                                                            pixels=1)
        obj_base, obj_right = dsl.object_transform_split_object_along_axis(obj=obj_base, cut_orientation=vertical,
                                                                           pixels=dsl.get_object_feature_size_x(
                                                                               obj_base) - 1)
        to_left = dsl.make_new_dimension2d(-1, 0)
        obj_bottom = dsl.object_transform_translate_by_distance(obj_bottom, to_left)
        obj_right = dsl.object_transform_translate_by_distance(obj_right, to_left)
        out_canvas = dsl.add_object_to_canvas(out_canvas, obj_bottom)
        out_canvas = dsl.add_object_to_canvas(out_canvas, obj_right)
        out_canvas = dsl.add_object_to_canvas(out_canvas, obj_base)
    return out_canvas



def logic_function_045e512c(in_canvas: Canvas) -> Canvas:
    out_canvas = copy(in_canvas)
    largest_object = dsl.select_largest_object_by_area(out_canvas)
    other_objects = dsl.select_rest_of_the_objects(canvas=out_canvas, obj=largest_object)
    for oo in other_objects:
        point = dsl.get_point_for_match_shape_furthest(largest_object, oo, match_shape_only=True,
                                                       try_unique=False, transformations=[],
                                                       padding=Surround(0, 0, 0, 0))
        canvas_pos_oo = dsl.get_object_feature_canvas_pos(oo)
        dist, _ = dsl.furthest_point_to_point(point, canvas_pos_oo)
        colour = dsl.get_object_feature_colour(oo)
        obj = dsl.copy_object(largest_object)
        for _ in range(3):
            obj = dsl.object_transform_new_colour(obj, colour)
            obj = dsl.object_transform_translate_along_direction(obj, dist)
            out_canvas = dsl.add_object_to_canvas(out_canvas, obj)

    return out_canvas

#TODO: This is a code logic.
'''
def logic_function_0520fde7(in_canvas: Canvas) -> Canvas:
    out_canvas = dsl.make_new_canvas(size=Dimension2D(dsl.get_canvas_feature_size(in_canvas).dy,
                                                      dsl.get_canvas_feature_size(in_canvas).dy))
    randoms = dsl.select_objects_of_type(in_canvas, Random)
    result_object = dsl.canvas_transform_and_objects(in_canvas, randoms[0], randoms[1],
                                                     new_colour=result_colour)
    out_canvas = dsl.add_object_to_canvas(out_canvas, result_object)

    return out_canvas
'''

def logic_function_05269061(in_canvas: Canvas) -> Canvas:
    out_canvas = dsl.make_new_canvas_as(in_canvas)
    canvas_height = dsl.get_canvas_feature_size_y(in_canvas)
    objects = dsl.select_all_objects(in_canvas)
    num_of_iters = dsl.divide_to_int(dsl.get_canvas_feature_size_y(in_canvas), 3)

    for o in objects:
        colour = dsl.get_object_feature_colour(o)
        canvas_pos = dsl.get_object_feature_canvas_pos(o)
        if dsl.get_object_feature_canvas_pos_x(o) != 0:
            mult = dsl.modulo_point_x(canvas_pos, 3)
            if dsl.equal(mult, 1):
                mult = dsl.assign(2)
            elif dsl.equal(mult, 2):
                mult = dsl.assign(1)
        else:
            mult = dsl.modulo_point_y(canvas_pos, 3)
        for i in range(-num_of_iters - 1, num_of_iters + 1):
            d = dsl.make_new_diagonal(height=canvas_height, colour=colour, canvas_pos=Point(0, i * 3 + mult))
            out_canvas = dsl.add_object_to_canvas(out_canvas, d)
    return out_canvas



def logic_function_05f2a901_original(in_canvas: Canvas) -> Canvas | None:
    try:
        out_canvas = dsl.make_new_canvas_as(in_canvas)
        large = dsl.select_only_object_of_colour(in_canvas, colour=random_colours[0].index)
        small = dsl.select_only_object_of_colour(in_canvas, colour=random_colours[1].index)
        dist = dsl.get_distance_touching_between_objects(small, large)

        updated_small = dsl.object_transform_translate_along_direction(small, dist)
        out_canvas = dsl.add_object_to_canvas(out_canvas, updated_small)
        out_canvas = dsl.add_object_to_canvas(out_canvas, large)

        return out_canvas
    except:
        return None



def logic_function_06df4c85(in_canvas: Canvas) -> Canvas | None:
    try:
        out_canvas = dsl.copy_canvas(in_canvas)
        object_colours = in_canvas.get_used_colours()
        for c in object_colours:
            objects = dsl.select_all_objects_of_colour(in_canvas, c)
            pos = []
            for o in objects:
                cp = dsl.get_object_feature_canvas_pos(o)
                tile_xy = dsl.get_tile_from_canvas_pos(in_canvas, cp)
                pos.append(dsl.tuple_to_point(tile_xy))
            pairs = dsl.all_binary_combinations(pos)

            for p in pairs:
                points_to_fil = dsl.all_points_between_two_points(p[0], p[1], cardinal_only=True)
                for cp in points_to_fil:
                    cp = dsl.point_to_tuple(cp)
                    cp = dsl.get_canvas_pos_from_tile(in_canvas, cp)
                    tile_size = dsl.get_canvas_feature_grid_tile_size(in_canvas)
                    new_object = dsl.make_new_parallelogram(size=Dimension2D(tile_size, tile_size),
                                                            canvas_pos=cp, colour=c)
                    out_canvas = dsl.add_object_to_canvas(out_canvas, new_object)

        return out_canvas
    except:
        return None


# Note. This is a CODE task. That means that the logic can not be correctly generated within only a single input canvas
# For the generator, this extra info comes ready made from outside the in_canvas.

def logic_function_08ed6ac7(in_canvas: Canvas) -> Canvas | None:
    try:
        out_canvas = dsl.make_new_canvas_as(in_canvas)
        objects = dsl.select_all_objects(in_canvas)
        ordered = dsl.order_objects_over_height(objects, reverse=False)
        for i, o in enumerate(ordered):
            o = dsl.object_transform_new_colour(o, colours_in_order[i])
            out_canvas = dsl.add_object_to_canvas(out_canvas, o)

        return out_canvas
    except:
        return None


def logic_function_09629e4f(in_canvas: Canvas) -> Canvas | None:
    try:
        out_canvas = dsl.make_new_canvas_as(in_canvas)
        good_object = dsl.select_object_with_the_fewer_colours(in_canvas)
        good_object = dsl.object_transform_translate_to_point(good_object, target_point=Point(0, 0))

        c_pos = dsl.get_object_feature_coloured_positions(good_object)
        for p in c_pos:
            colour = dsl.get_object_feature_colour_at_position(good_object, p)
            canvas_pos = dsl.get_canvas_pos_from_tile(in_canvas, (p.x, p.y))
            size = dsl.get_object_feature_size(good_object)
            o = dsl.make_new_parallelogram(size=size, colour=colour, canvas_pos=canvas_pos)
            out_canvas = dsl.add_object_to_canvas(out_canvas, o)

        return out_canvas
    except:
        return None


def logic_function_0962bcdd(in_canvas: Canvas) -> Canvas | None:
    try:
        out_canvas = dsl.make_new_canvas_as(in_canvas)
        objects = dsl.select_all_objects(in_canvas)
        for o in objects:
            center_pos = dsl.get_object_feature_relative_point_position(o, RelativePoint.Middle_Center)
            center_col = dsl.get_object_feature_colour_at_position(o, center_pos)
            col = dsl.get_object_feature_colour(o)
            c = dsl.make_new_cross(size=Dimension2D(5, 5), colour=col)
            c = dsl.object_transform_translate_to_point(c, target_point=center_pos, object_point=Point(2, 2, 0))
            i = dsl.make_new_inverse_cross(height=5, colour=center_col)
            i = dsl.object_transform_translate_to_point(i, target_point=center_pos, object_point=Point(2, 2, -1))
            out_canvas = dsl.add_object_to_canvas(out_canvas, c)
            out_canvas = dsl.add_object_to_canvas(out_canvas, i)

        return out_canvas
    except:
        return None



def logic_function_0a938d79(in_canvas: Canvas) -> Canvas | None:
    try:
        out_canvas = dsl.make_new_canvas_as(in_canvas)
        objects = dsl.select_all_objects(in_canvas)
        horizontal_canvas = dsl.bigger_than(dsl.get_canvas_feature_size_x(in_canvas),
                                            dsl.get_canvas_feature_size_y(in_canvas))
        for o in objects:
            if horizontal_canvas:
                height = dsl.get_canvas_feature_size_y(in_canvas)
                dimensions = dsl.make_new_dimension2d(dx=1, dy=height)
                x_pos = dsl.get_object_feature_canvas_pos_x(obj=o)
                canvas_pos = dsl.make_new_point(x=x_pos, y=0)

                dist = dsl.get_along_x_distance_between_objects(objects[0], objects[1])
                times = dsl.divide_to_int(dsl.subtract(dsl.get_canvas_feature_size_x(in_canvas), x_pos),
                                          dsl.get_length_of_vector(dist))
            else:
                width = dsl.get_canvas_feature_size_x(in_canvas)
                dimensions = dsl.make_new_dimension2d(dx=width, dy=1)
                y_pos = dsl.get_object_feature_canvas_pos_y(o)
                canvas_pos = dsl.make_new_point(x=0, y=y_pos)

                dist = dsl.get_along_y_distance_between_objects(objects[0], objects[1])
                times = dsl.divide_to_int(dsl.get_canvas_feature_size_y(in_canvas),
                                          dsl.sum(dsl.get_length_of_vector(dist), 1))

            colour = dsl.get_object_feature_colour(obj=o)
            line = dsl.make_new_parallelogram(size=dimensions, colour=colour)

            line = dsl.object_transform_translate_to_point(obj=line, target_point=canvas_pos)
            out_canvas = dsl.add_object_to_canvas(canvas=out_canvas, obj=line)

            for _ in range(times):
                line = dsl.object_transform_translate_along_direction(line, dsl.multiply_vector(dist, 2))
                out_canvas = dsl.add_object_to_canvas(canvas=out_canvas, obj=line)

        return out_canvas
    except:
        return None


def logic_function_0b148d64(in_canvas: Canvas) -> Canvas | None:
    try:
        num_of_objects_per_colour, objects_per_colour = dsl.group_objects_according_to_colour(in_canvas)
        index = 1 if dsl.bigger_than(num_of_objects_per_colour[0], num_of_objects_per_colour[1]) else 0
        obj = dsl.select_from_list(dsl.select_from_list(objects_per_colour, index), 0)
        obj = dsl.object_transform_translate_to_point(obj, target_point=Point(0, 0))
        size = dsl.get_object_feature_size(obj)
        out_canvas = dsl.make_new_canvas(size)
        out_canvas = dsl.add_object_to_canvas(out_canvas, obj)

        return out_canvas
    except:
        return None

def logic_function_0ca9ddb6(in_canvas: Canvas) -> Canvas | None:
    try:
        out_canvas = dsl.copy_canvas(in_canvas)
        crosses_dots = dsl.select_all_objects_of_colour(in_canvas, in_colours[0])
        for cd in crosses_dots:
            canvas_pos = dsl.get_object_feature_canvas_pos(cd)
            c_canvas_pos = dsl.subtract_points(canvas_pos, Point(1, 1))
            cross = dsl.make_new_cross(size=Dimension2D(3, 3), colour=out_colours[0], canvas_pos=c_canvas_pos)
            cd = dsl.object_transform_change_depth(cd, 10)
            out_canvas = dsl.add_object_to_canvas(out_canvas, cross)
            out_canvas = dsl.add_object_to_canvas(out_canvas, cd)

        inv_crosses_dots = dsl.select_all_objects_of_colour(in_canvas, in_colours[1])
        for cd in inv_crosses_dots:
            canvas_pos = dsl.get_object_feature_canvas_pos(cd)
            c_canvas_pos = dsl.subtract_points(canvas_pos, Point(1, 1))
            inv_cross = dsl.make_new_inverse_cross(height=3, colour=out_colours[1], canvas_pos=c_canvas_pos)
            cd = dsl.object_transform_change_depth(cd, 10)
            out_canvas = dsl.add_object_to_canvas(out_canvas, inv_cross)
            out_canvas = dsl.add_object_to_canvas(out_canvas, cd)

        return out_canvas
    except:
        return None

def logic_function_0d3d703e(in_canvas: Canvas) -> Canvas | None:
    try:
        out_canvas = dsl.make_new_canvas_as(in_canvas)

        objects = dsl.select_all_objects(in_canvas)
        for o in objects:
            colour = dsl.get_object_feature_colour(o)
            index = dsl.index_of_item_in_list(in_colours, colour)
            new_colour = dsl.select_from_list(out_colours, index)
            height = dsl.get_canvas_feature_size_y(in_canvas)
            canvas_pos = dsl.get_object_feature_canvas_pos(o)
            line = dsl.make_new_parallelogram(size=Dimension2D(1, height),
                                              colour=new_colour, canvas_pos=canvas_pos)
            out_canvas = dsl.add_object_to_canvas(out_canvas, line)

        return out_canvas
    except:
        return None


def logic_function_0e206a2e(in_canvas: Canvas) -> Canvas | None:
    try:
        out_canvas = dsl.make_new_canvas_as(in_canvas)
        main_obj = dsl.select_object_with_the_most_colours(in_canvas)
        main_colour = dsl.get_object_feature_colour(main_obj)
        all_main_objects = dsl.select_all_objects_of_colour(in_canvas, main_colour)
        all_anchor_objects = dsl.select_rest_of_the_objects(in_canvas, all_main_objects)
        for main in all_main_objects:
            rot, trans = dsl.get_point_and_rotation_for_best_match_to_objects(object_to_move=main,
                                                                              target_objects=all_anchor_objects,
                                                                              match_shape_only=False)
            main = dsl.object_transform_rotate(main, rot)
            main = dsl.object_transform_translate_to_point(main, trans)
            out_canvas = dsl.add_object_to_canvas(out_canvas, main)
        return out_canvas
    except:
        return None

def logic_function_10fcaaa3(in_canvas: Canvas) -> Canvas | None:
    try:
        in_canvas_size = dsl.get_canvas_feature_size(in_canvas)
        out_canvas_size = dsl.multiply_dimension(in_canvas_size, 2)
        out_canvas = dsl.make_new_canvas(size=out_canvas_size)

        objects = dsl.select_all_objects(in_canvas)

        for obj in objects:
            cp = dsl.get_object_feature_canvas_pos(obj)
            cp = dsl.subtract_points(cp, Point(1, 1))

            new_obj_0 = dsl.make_new_inverse_cross(height=3, canvas_pos=cp, colour=pattern_colour)
            out_canvas = dsl.add_object_to_canvas(out_canvas, new_obj_0)

            new_vect = dsl.make_new_vector(orientation=Orientation.Right,
                                           length=dsl.get_canvas_feature_size_x(in_canvas),
                                           origin=cp)
            new_obj_right = dsl.object_transform_translate_along_direction(new_obj_0, new_vect)
            new_cp = dsl.get_object_feature_canvas_pos(new_obj_right)
            out_canvas = dsl.add_object_to_canvas(out_canvas, new_obj_right)

            new_vect = dsl.make_new_vector(orientation=Orientation.Up,
                                           length=dsl.get_canvas_feature_size_y(in_canvas),
                                           origin=new_cp)
            new_obj_upright = dsl.object_transform_translate_along_direction(new_obj_right, new_vect)
            out_canvas = dsl.add_object_to_canvas(out_canvas, new_obj_upright)

            new_vect = dsl.make_new_vector(orientation=Orientation.Up,
                                           length=dsl.get_canvas_feature_size_y(in_canvas),
                                           origin=cp)
            new_obj_up = dsl.object_transform_translate_along_direction(new_obj_0, new_vect)
            out_canvas = dsl.add_object_to_canvas(out_canvas, new_obj_up)

            cp = dsl.sum_points(cp, Point(1, 1, 10))
            new_obj_0 = dsl.object_transform_translate_to_point(obj, cp)
            out_canvas = dsl.add_object_to_canvas(out_canvas, new_obj_0)

            new_vect = dsl.make_new_vector(orientation=Orientation.Right,
                                           length=dsl.get_canvas_feature_size_x(in_canvas),
                                           origin=cp)
            new_obj_right = dsl.object_transform_translate_along_direction(obj, new_vect)
            new_obj_right = dsl.object_transform_translate_to_front_of_all(out_canvas, new_obj_right)
            new_cp = dsl.get_object_feature_canvas_pos(new_obj_right)
            out_canvas = dsl.add_object_to_canvas(out_canvas, new_obj_right)

            new_vect = dsl.make_new_vector(orientation=Orientation.Up,
                                           length=dsl.get_canvas_feature_size_y(in_canvas),
                                           origin=new_cp)
            new_obj_upright = dsl.object_transform_translate_along_direction(new_obj_right, new_vect)
            new_obj_upright = dsl.object_transform_translate_to_front_of_all(out_canvas, new_obj_upright)
            out_canvas = dsl.add_object_to_canvas(out_canvas, new_obj_upright)

            new_vect = dsl.make_new_vector(orientation=Orientation.Up,
                                           length=dsl.get_canvas_feature_size_y(in_canvas),
                                           origin=cp)
            new_obj_up = dsl.object_transform_translate_along_direction(new_obj_0, new_vect)
            new_obj_up = dsl.object_transform_translate_to_front_of_all(out_canvas, new_obj_up)
            out_canvas = dsl.add_object_to_canvas(out_canvas, new_obj_up)

        return out_canvas
    except:
        return None

def logic_function_11852cab(in_canvas: Canvas) -> Canvas | None:
    try:
        out_canvas = dsl.make_new_canvas_as(in_canvas)
        objects = dsl.select_all_objects(in_canvas)
        for obj in objects:
            out_canvas = dsl.add_object_to_canvas(out_canvas, obj)
            for i in range(4):
                o = dsl.object_transform_rotate(obj, i)
                out_canvas = dsl.add_object_to_canvas(out_canvas, o)
        return out_canvas
    except:
        return None


def logic_function_1190e5a7(in_canvas: Canvas) -> Canvas | None:
    try:
        large = dsl.select_largest_object_by_area(in_canvas)
        others = dsl.select_rest_of_the_objects(in_canvas, large)
        colour = dsl.get_object_feature_colour(large)
        x = dsl.assign(1)
        y = dsl.assign(1)
        for o in others:
            if dsl.bigger_than_or_equal(dsl.get_object_feature_canvas_pos_x(o), 1):
                x = dsl.sum(x, 1)
            if dsl.bigger_than_or_equal(dsl.get_object_feature_canvas_pos_y(o), 1):
                y = dsl.sum(y, 1)
        obj = dsl.make_new_parallelogram(size=Dimension2D(x, y), colour=colour)

        out_canvas = dsl.make_new_canvas(size=Dimension2D(x, y))
        out_canvas = dsl.add_object_to_canvas(out_canvas, obj)

        return out_canvas
    except:
        return None


def logic_function_137eaa0f(in_canvas: Canvas) -> Canvas | None:
    try:

        largest = dsl.select_largest_object_by_area(in_canvas)
        colour_large = dsl.get_object_feature_least_used_colour(largest)
        smallest = dsl.select_smallest_object_by_area(in_canvas)
        colour_small = dsl.get_object_feature_least_used_colour(smallest)
        colour = dsl.select_from_list(dsl.intersect(colour_small, colour_large), 0)
        target_pos = dsl.get_object_feature_position_of_colour(largest, colour)
        others = dsl.select_rest_of_the_objects(in_canvas, largest)

        largest = dsl.object_transform_translate_to_point(largest, Point(0, 0, 0))
        for o in others:
            dsl.get_object_feature_coloured_positions(o)
            pos = dsl.get_object_feature_position_of_colour(o, colour)
            o = dsl.object_transform_translate_to_point(o, target_point=target_pos, object_point=pos)
            largest = dsl.object_transform_add_two_objects(largest, o)

        size = dsl.get_object_feature_size(largest)
        out_canvas = dsl.make_new_canvas(size=size)
        out_canvas = dsl.add_object_to_canvas(out_canvas, largest)

        return out_canvas
    except:
        return None

def logic_function_150deff5(in_canvas: Canvas) -> Canvas | None:
    try:
        out_canvas = dsl.make_new_canvas_as(in_canvas)

        all_objects = dsl.select_all_objects(in_canvas)
        large_area = dsl.assign(0)
        for o in all_objects:
            obj_area = dsl.multiply(dsl.get_object_feature_size_x(o), dsl.get_object_feature_size_x(o))
            if dsl.bigger_than(obj_area, large_area):
                large_area = obj_area

        for o in all_objects:
            obj_area = dsl.multiply(dsl.get_object_feature_size_x(o), dsl.get_object_feature_size_x(o))
            if obj_area == large_area:
                o = dsl.object_transform_new_colour(obj=o, colour=colours[1])
            else:
                o = dsl.object_transform_new_colour(obj=o, colour=colours[2])

            out_canvas = dsl.add_object_to_canvas(out_canvas, o)

        return out_canvas
    except:
        return None


def logic_function_b775ac94(in_canvas: Canvas) -> Canvas | None:
    try:
        object_list = dsl.select_rest_of_the_objects(in_canvas, obj=None)
        canvas_out = dsl.copy_canvas(in_canvas)

        for init_obj in object_list:
            init_obj = dsl.object_transform_change_depth(init_obj, 0)
            canvas_temp = dsl.make_new_canvas_as(in_canvas)
            canvas_temp = dsl.add_object_to_canvas(canvas_temp, init_obj)
            canvas_temp = dsl.canvas_transform_split_object_by_colour_on_canvas(canvas_temp, init_obj)
            largest_object = dsl.select_largest_object_by_area(canvas_temp)
            other_objects = dsl.select_rest_of_the_objects(canvas_temp, largest_object)

            for oo in other_objects:
                colour = dsl.get_object_feature_colour(oo)
                dist = dsl.get_distance_min_between_objects(largest_object, oo)
                largest_object = dsl.object_transform_flip_and_translate(largest_object, dist)
                largest_object = dsl.object_transform_new_colour(largest_object, colour)
                canvas_out = dsl.add_object_to_canvas(canvas_out, largest_object)
        return canvas_out
    except:
        return None
# </editor-fold>


