
from copy import copy
from typing import List, Tuple, Any, Type
from itertools import combinations
import numpy as np

from structure.canvas.canvas import Canvas
from structure.geometry.basic_geometry import Point, Vector, Dimension2D, Surround, RelativePoint, Orientation, Colour
from structure.object.object import Object
from structure.object.primitives import Primitive, Predefined, Random, Parallelogram, Cross, Hole, Pi, InverseCross, \
    Dot, Angle, Diagonal, Steps, Fish, Bolt, Tie, Spiral, Pyramid, Maze
from structure.task.task import Task


# <editor-fold desc="Math functions">
def assign(a: int | float | bool | Point | Dimension2D | Primitive | Vector| Surround) -> \
        int | float | bool | Point | Dimension2D | Primitive | Vector| Surround:
    """
    Return a
    :param a: a ->  int | float | bool | Point | Dimension2D | Primitive | Vector| Surround
    :return: a -> int | float | bool | Point | Dimension2D | Primitive | Vector| Surround
    """
    return a


def summation(a: int | float, b: int | float) -> int | float:
    """
    Adds a and b
    :param a: a -> int | float
    :param b: b -> int | float
    :return: a + b -> int | float
    """
    return a + b


def subtraction(a: int | float, b: int | float) -> int | float:
    """
    Subtract b from a
    :param a: a -> int | float
    :param b: b -> int | float
    :return: a - b -> int | float
    """
    return a - b


def multiplication(a: int | float, b: int | float) -> int | float:
    """
    Multiply a with b
    :param a: a -> int | float
    :param b: b -> int | float
    :return: a * b -> int | float
    """
    return a * b


def division(a: int | float, b: int | float) -> int | float:
    """
    Divide a by b
    :param a: a -> int | float
    :param b: b -> int | float
    :return: a / b -> int | float
    """
    return a / b


def divide_to_int(a: int | float, b: int | float) -> int | float:
    """
    Divide a by b and floor to nearest int
    :param a: a -> int | float
    :param b: b -> int | float
    :return: a//b -> int | float
    """
    return a // b


def modulo(a: int, b: int) -> int:
    """
    Returns the quotient of a / b
    :param a: a -> int
    :param b: b -> int
    :return: a % b -> int
    """
    return a % b


def sign(a: int | float) -> int:
    """
    Get the sign of a
    :param a: a -> int | float
    :return: sign(a)  -> int
    """
    return np.sign(a)


def bigger_than(a: int | float, b: int | float) -> bool:
    """
    Return True if a is bigger than b else False
    :param a: a -> int | float
    :param b: b -> int | float
    :return: a > b -> bool
    """
    return a > b


def bigger_than_or_equal(a: int | float, b: int | float) -> bool:
    """
    Return True if a is bigger than or equal to b else False
    :param a: a -> int | float
    :param b: b -> int | float
    :return: a >= b -> bool
    """
    return a >= b


def equal(a: int | float | Point | Dimension2D | Vector, b: int | float | Point | Dimension2D | Vector) -> bool:
    """
    Return True if a is equal to b else False. This works not only for ints and floats but for Points, Dimension2Ds and Vectors
    :param a: a -> int | float | Point | Dimension2D | Vector
    :param b: b -> int | float | Point | Dimension2D | Vector
    :return: a > b -> float
    """
    return a == b


def not_equal(a: int | float | Point | Dimension2D | Vector, b: int | float | Point | Dimension2D | Vector) -> bool:
    """
    Return True if a is not equal to b else False. This works not only for ints and floats but for Points, Dimension2Ds and Vectors
    :param a: a -> int | float | Point | Dimension2D | Vector
    :param b: b -> int | float | Point | Dimension2D | Vector
    :return: a > b -> float
    """
    return a != b


def all_binary_combinations(array: List[Any]) -> List[Tuple[Any, Any]]:
    """
    Returns all possible pair combinations of the positions of the elements of a list.
    :param array: The list of elements -> List[Any]
    :return: A list of tuples, each tuple is a combination of two elements from the list -> List[Tuple[Any]]
    """
    return list(combinations(array, 2))


def intersection(a: Any, b: Any) -> List:
    """
    Gets the common elements in lists a and b. If a and, or b are not iterables they are turned to single element lists.
    :param a: a -> Any
    :param b: b -> Any
    :return: The intersection of a and b -> List
    """
    a = a if isinstance(a, list) else [a]
    b = a if isinstance(b, list) else [b]
    result = set(a).intersection(set(b))

    return list(result)

def difference(a: Any, b: Any) -> List:
    """
    Gets the non common elements in lists a and b. If a and, or b are not iterables they are turned to single element lists.
    :param a: a -> Any
    :param b: b -> Any
    :return: The intersection of a and b -> List
    """
    a = a if isinstance(a, list) else [a]
    b = a if isinstance(b, list) else [b]
    result = set(a).difference(set(b))

    return list(result)

def union(a: Any, b: Any) -> List:
    """
    Gets the union (all elements from both lists) of a and b. If a and, or b are not iterables they are turned to single element lists.
    :param a: a -> Any
    :param b: b -> Any
    :return: The intersection of a and b -> List
    """
    a = a if isinstance(a, list) else [a]
    b = a if isinstance(b, list) else [b]
    result = set(a).union(set(b))

    return list(result)
# </editor-fold>

# <editor-fold desc="Functions on List selection)">
def get_item_n_from_list(list: List[Any], n: int) -> Any:
    """
    Get the index item of the list
    :param list: The list -> List[Any]
    :param n: The index -> int
    :return: The item list[n] -> Any
    """
    return copy(list[n])

def replace_n_item_in_list(list: List[Any], n: int, item: Any) -> Any:
    """
    Get a list where the nth item is replaces by item
    :param list: The list -> List[Any]
    :param n: The position index -> int
    :param item: The new item -> Any
    :return: The updated list  -> List[Any]
    """
    list[n] = item
    return [copy(o) for o in list]

def index_of_first_item_in_list(list: List, value: Any) -> int | None:
    """
    Get the index n of the first time the value appears in the list
    :param list: The list
    :param value: The value to look for
    :return: The index of the value in the list
    """
    try:
        return list.index(value)
    except ValueError:
        return None

def get_first_item(list: List[Any]) -> Any:
    """
    Gets the first item in the list
    :param list: The list
    :return: The first item
    """
    return copy(list[0])

def get_last_item(list: List[Any]) -> Any:
    """
    Gets the last item in the list
    :param list: The list
    :return: The last item
    """
    return copy(list[-1])

def get_all_but_first_item(list: List[Any]) -> Any:
    """
    Gets the all but the first item in the list
    :param list: The list
    :return: All but the first item
    """
    return [copy(i) for i in list[1:]]

def get_all_but_last_item(list: List[Any]) -> Any:
    """
    Gets the all but the last item in the list
    :param list: The list
    :return: All but the last item
    """
    return [copy(i) for i in list[:-1]]

def length_of_list(list: List[Any]) -> int:
    """
    Gets the length of the list
    :param list: The list
    :return: The length of the list
    """
    return len(list)

def arange(size: int, start: int = 0, step: int = 1) -> range:
    """
    Returns the range of length size starting at start and stepping step
    :param size: The length of the range -> int
    :param start: The starting integer -> int
    :param step: The stepping amount -> int
    :return: A range -> range
    """
    return range(start, size, step)
# </editor-fold>

# <editor-fold desc="Functions on Structure (Points, Distance2D, Vector, etc)">
def make_new_point(x: int, y: int, z:int = 0) -> Point:
    """
    Create a new Point from x, y, z integers. If z is not given it defauls to 0.
    :param x: The x coordinate -> int
    :param y: The y coordinate -> int
    :param z: The z coordinate -> int
    :return: The Point (x,y,z) -> Point
    """
    return Point(x, y, z)


def make_new_dimension2d(dx: int, dy: int) -> Dimension2D:
    """
    Create a new Dimension2D from dx (width) and dy (height) integers
    :param dx: Width -> int
    :param dy: Height -> int
    :return: The nw Dimension2D -> Dimension2D
    """
    return Dimension2D(dx, dy)


def make_new_orientation(towards: str | int) -> Orientation:
    if isinstance(towards, str):
        where = {'Up': 0, 'Up_Right': 1, 'Right': 2, 'Down_Right': 3, 'Down': 4, 'Down_Left': 5, 'Left': 6, 'Up_Left': 7}[towards]
    else:
        where = towards
    return Orientation(where)


def tuple_to_point(t: Tuple[int, int]) -> Point:
    return Point(t[0], t[1])


def point_to_tuple(p: Point) -> Tuple[int, int]:
    return p.x, p.y


def furthest_point_to_point(origin: Point, targets: List[Point] | Point) -> Tuple[Vector, int]:
    if type(targets) == Point:
        return origin.euclidean_distance(targets), 0
    result = targets[0]
    index = 0
    for i, t in enumerate(targets):
        if origin.euclidean_distance(result) is None:
            result = t
            index = i
        elif origin.euclidean_distance(t) is not None:
            if origin.euclidean_distance(t).length > origin.euclidean_distance(result).length:
                result = t
                index = i

    return origin.euclidean_distance(result), index


def closest_point_to_point(origin: Point, targets: List[Point] | Point) -> Tuple[Vector, int]:
    if type(targets) == Point:
        return origin.euclidean_distance(targets), 0
    result = targets[0]
    index = 0
    for i, t in enumerate(targets):
        if origin.euclidean_distance(result) is None:
            result = t
            index = i
        elif origin.euclidean_distance(t) is not None:
            if origin.euclidean_distance(t).length < origin.euclidean_distance(result).length:
                result = t
                index = i

    return origin.euclidean_distance(result), index


def sum_points(first: Point, second: Point) -> Point:
    """
    Add two Points (first.x + second.x, first.y + second.y)
    :param first: The first Point -> Point
    :param second: The second Point -> Point
    :return: The first + second Point -> Point
    """
    f = copy(first)
    s = copy(second)
    return f + s


def subtract_points(first: Point, second: Point) -> Point:
    """
    Subtract two Points (first.x - second.x, first.y - second.y)
    :param first: The first Point -> Point
    :param second: The second Point -> Point
    :return: The first - second Point -> Point
    """
    f = copy(first)
    s = copy(second)
    return f - s


def multiply_point(point: Point, mult: int | Point) -> Point:
    return copy(point) * mult


def multiply_point_x(point: Point, mult: int | Point) -> Point:
    return Point(point.x * mult, point.y)


def multiply_point_y(point: Point, mult: int | Point) -> Point:
    return Point(point.x, point.y * mult)


def mat_multiply_point(point: Point, other: Point | Dimension2D) -> Point:
    other = copy(other)
    other = Point(other.dx, other.dy) if isinstance(other, Dimension2D) else other
    return copy(point) @ copy(other)


def sum_dimensions(dimension: Dimension2D, other: Dimension2D) -> Dimension2D:
    return dimension + other


def multiply_dimension_x(dimension: Dimension2D, mult: int | float) -> Dimension2D:
    return Dimension2D(int(dimension.dx * mult), dimension.dy)


def multiply_dimension_y(dimension: Dimension2D, mult: int | float) -> Dimension2D:
    return Dimension2D(dimension.dx, int(dimension.dy * mult))


def multiply_dimension(dimension: Dimension2D, mult: int | float) -> Dimension2D:
    return Dimension2D(int(dimension.dx * mult), int(dimension.dy * mult))


def mat_multiply_dimension(dimension: Dimension2D, other: Dimension2D) -> Dimension2D:
    return copy(dimension) @ copy(other)


def modulo_point(point: Point, divisor: int) -> Point:
    return Point(point.x % divisor, point.y % divisor)


def modulo_point_x(point: Point, divisor: int) -> int:
    return point.x % divisor


def modulo_point_y(point: Point, divisor: int) -> int:
    return point.y % divisor


def points_in_line(start_point: Point, end_point: Point, cardinal_only: bool = False) -> Orientation | None:
    if start_point.x == end_point.x:
        if start_point.y > end_point.y:
            return Orientation.Down
        elif end_point.y > start_point.y:
            return Orientation.Up
    elif start_point.y == end_point.y:
        if start_point.x > end_point.x:
            return Orientation.Left
        elif start_point.x < end_point.x:
            return Orientation.Right
    if not cardinal_only:
        if np.abs(start_point.x - end_point.x) == np.abs(start_point.y - end_point.y):
            if start_point.x > end_point.x and start_point.y > end_point.y:
                return Orientation.Down_Left
            elif start_point.x < end_point.x and start_point.y > end_point.y:
                return Orientation.Down_Right
            elif start_point.x < end_point.x and start_point.y < end_point.y:
                return Orientation.Up_Right
            elif start_point.x > end_point.x and start_point.y < end_point.y:
                return Orientation.Up_Left

    return None


def all_points_between_two_points(start_point: Point, end_point: Point, cardinal_only: bool = False) -> List[Point]:
    all_points_in_line = []
    dir = points_in_line(start_point, end_point, cardinal_only)
    if dir is not None:
        if dir == Orientation.Up:
            for i in range(start_point.y, end_point.y + 1):
                all_points_in_line.append(Point(start_point.x, i, start_point.z))
        if dir == Orientation.Down:
            for i in range(start_point.y, end_point.y - 1, -1):
                all_points_in_line.append(Point(start_point.x, i, start_point.z))
        if dir == Orientation.Left:
            for i in range(start_point.x, end_point.x - 1, - 1):
                all_points_in_line.append(Point(i, start_point.y, start_point.z))
        if dir == Orientation.Right:
            for i in range(start_point.x, end_point.x + 1):
                all_points_in_line.append(Point(i, start_point.y, start_point.z))

        if not cardinal_only:
            if dir == Orientation.Up_Left:
                for i in range(0, end_point.y - start_point.y + 1):
                    all_points_in_line.append(Point(start_point.x - i, start_point.y + i))
            if dir == Orientation.Up_Right:
                for i in range(0, end_point.y - start_point.y + 1):
                    all_points_in_line.append(Point(start_point.x + i, start_point.y + i))
            if dir == Orientation.Down_Left:
                for i in range(0, start_point.y - end_point.y + 1):
                    all_points_in_line.append(Point(start_point.x - i, start_point.y - i))
            if dir == Orientation.Down_Right:
                for i in range(0, start_point.y - end_point.y + 1):
                    all_points_in_line.append(Point(start_point.x + i, start_point.y - i))

    return all_points_in_line


def make_new_vector(orientation: Orientation, length: int, origin: Point) -> Vector:
    return Vector(orientation=orientation, length=length, origin=origin)


def get_length_of_vector(v: Vector) -> int:
    return v.length


def get_orientation_of_vector(v: Vector) -> Orientation:
    return v.orientation


def get_origin_of_vector(v: Vector) -> Point:
    return v.origin


def multiply_vector(v: Vector, mult: int) -> Vector:
    return v * mult
# </editor-fold>

# <editor-fold desc="Functions to generate Objects">
def generate_contiguous_colour_objects(canvas: Canvas) -> Canvas:
    """
    Takes the actual_pixels of the Canvas object and generates Objects from contiguous coloured pixels. So all touching
    pixels with a specific colour become an Object. The generated Objects are added to the returned Canvas.
    :param canvas: The Canvas with coloured actual_pixels array that needs to be broken into objects.
    :return: The Canvas with the new Objects.
    """
    canvas.generate_contiguous_objects_by_colour()
    return copy(canvas)

def generate_contiguous_colour_objects_on_all_canvases(task : Task) -> Task:
    task.generate_contiguous_objects_by_colour()
    result = copy(task)
    return result

def generate_contiguous_objects(canvas: Canvas) -> Canvas:
    """
    Takes the actual_pixels of the Canvas object and generates Objects from contiguous pixels irrespective of their colour.
    So all touching pixels no matter what their colour become an Object. The generated Objects are added to the returned Canvas.
    :param canvas: The Canvas with coloured actual_pixels array that needs to be broken into objects.
    :return: The Canvas with the new Objects.
    """
    temp_canvas = copy(canvas)
    temp_canvas.generate_contiguous_objects()
    return temp_canvas
# </editor-fold>


# <editor-fold desc="Functions on Canvasses">
def copy_canvas(canvas: Canvas) -> Canvas:
    return copy(canvas)


def copy_object(obj: Primitive) -> Primitive:
    return copy(obj)


def make_new_canvas_as(canvas: Canvas) -> Canvas:
    """
    Creates a new Canvas (no Objects and actual_pixels = 1) with the same size as canvas. It also copies onto the new Canvas
    any grid that canvas might have.
    :param canvas: The Canvas to copy the size of -> Canvas
    :return: The new Canvas -> Canvas
    """
    if canvas.grid:
        new_canvas = Canvas(as_grid_x_y_tilesize_colour=(canvas.grid_shape[0], canvas.grid_shape[1],
                                                         canvas.size_of_tiles, canvas.grid_colour))
    else:
        new_canvas = Canvas(size=canvas.size)
    return new_canvas


def make_new_canvas(size: Dimension2D) -> Canvas:
    """
    Creates a new Canvas (no Objects and actual_pixels = 1) with Canvas.size = size.
    :param size: The size of the Canvas -> Dimension2D
    :return: The new Canvas -> Canvas
    """
    return Canvas(size=size)

def get_all_objects_in_canvas(canvas: Canvas) -> List[Primitive]:
    """
    Returns all the Objects in the canvas as a List
    :param canvas: The Canvas to get the Objects from -> Canvas
    :return: A list of Objects in the Canvas -> List[Object]
    """
    return [copy(o) for o in canvas.objects]

def get_canvas_feature_size(canvas: Canvas) -> Dimension2D:
    """
    Returns the size of the Canvas as a Dimension2D
    :param canvas: The Canvas to get the size of -> Canvas
    :return: The size of the Canvas -> Dimension2D
    """
    return canvas.size


def get_canvas_feature_size_x(canvas: Canvas) -> int:
    """
    Returns the x element (width) of the size of the Canvas as an integer
    :param canvas: The Canvas to get the size of -> Canvas
    :return: The width of the Canvas -> int
    """
    return canvas.size.dx


def get_canvas_feature_size_y(canvas: Canvas) -> int:
    """
    Returns the y element (height) of the size of the Canvas as an integer
    :param canvas: The Canvas to get the size of -> Canvas
    :return: The height of the Canvas -> int
    """
    return canvas.size.dy


def get_canvas_feature_all_object_colours(canvas: Canvas) -> List[int]:
    return canvas.get_used_colours()


def get_canvas_feature_grid_colour(canvas: Canvas) -> int | None:
    if canvas.grid:
        return canvas.grid_colour
    return None


def get_canvas_feature_grid_tile_size(canvas: Canvas) -> int | None:
    if canvas.grid:
        return canvas.size_of_tiles
    return None


def get_colour_common_to_all_objects(canvas: Canvas) -> int | List[int] | None:
    common_colours = list(canvas.objects[0].get_used_colours())
    for o in canvas.objects[1:]:
        common_colours = list(np.intersect1d(common_colours, list(o.get_used_colours())))

    if len(common_colours) == 0:
        return None
    elif len(common_colours) == 1:
        return common_colours[0]

    return common_colours


def add_object_to_canvas(canvas: Canvas, obj: Object) -> Canvas:
    """
    Adds the obj Object on the canvas Canvas (and makes sure it becomes part of the actual_pixels of the canvas)
    :param canvas: The Canvas to add the obj to -> Canvas
    :param obj: The Object to add -> Object
    :return: The Canvas with the new Object added -> Canvas
    """
    new_canvas = copy(canvas)
    new_obj = copy(obj)
    new_canvas.add_new_object(new_obj)
    return new_canvas


def canvas_transform_split_object_by_colour_on_canvas(canvas: Canvas, obj: Primitive) -> Canvas:
    new_canvas = copy(canvas)
    new_canvas.split_object_by_colour(obj)
    return new_canvas


def canvas_transform_and_objects(canvas: Canvas, obj_a: Primitive, obj_b: Primitive, new_colour: Colour,
                                 canvas_pos: Point = Point(0, 0)) -> Primitive:
    canvas = copy(canvas)
    new_object = canvas.and_objects(obj_a, obj_b, new_colour, canvas_pos)

    return new_object


def get_tile_from_canvas_pos(canvas: Canvas, pixel: Point) -> Tuple[int, int] | None:
    for k in canvas.grid_tiles_coordinates:
        point = Point(pixel.x, pixel.y, 0)
        if canvas.grid_tiles_coordinates[k] == point:
            return k
    return None


def get_canvas_pos_from_tile(canvas: Canvas, tile: Tuple[int, int]) -> Point:
    return canvas.grid_tiles_coordinates[tile]

def resize_canvas(canvas: Canvas, new_size: Dimension2D) -> Canvas:
    """
    Resizes the Canvas to new_size
    :param canvas: The Canvas to resize -> Canvas
    :param new_size: The new size -> Dimension2D
    :return: The resized Canvas -> Canvas
    """
    objects = canvas.objects
    canvas = make_new_canvas(new_size)
    for o in objects:
        canvas = add_object_to_canvas(canvas, copy(o))
    return canvas
# </editor-fold>


# <editor-fold desc="Functions on Task">
def select_input_test_canvas(task: Task, test_id: int) -> Canvas:
    task.test_input_canvases[test_id]
    return copy(task)
# </editor-fold>


# <editor-fold desc="Functions to get Primitive features">
def is_of_type(obj: Primitive, primitive_type: Any) -> bool:
    return type(obj) == primitive_type


def get_distance_min_between_objects(first: Primitive, second: Primitive) -> Vector | None:
    return first.get_distance_to_object(other=second, dist_type='min')


def get_distance_max_between_objects(first: Primitive, second: Primitive) -> Vector | None:
    return first.get_distance_to_object(other=second, dist_type='max')


def get_distance_origin_to_origin_between_objects(first: Primitive, second: Primitive) -> Vector | None:
    return first.get_distance_to_object(other=second, dist_type='canvas_pos')


def get_distance_touching_between_objects(first: Primitive, second: Primitive) -> Vector | None:
    dist = first.get_distance_to_object(other=second, dist_type='straight_line')
    if dist is not None:
        dist.length -= 1
        return dist
    return None


def get_along_x_distance_between_objects(first: Primitive, second: Primitive) -> Vector:
    orientation = Orientation.Left if first.canvas_pos.x > second.canvas_pos.x else Orientation.Right
    origin = first.canvas_pos
    length = np.abs(second.canvas_pos.x - first.canvas_pos.x).astype(int)
    return Vector(orientation=orientation, length=length, origin=origin)


def get_along_y_distance_between_objects(first: Primitive, second: Primitive) -> Vector:
    orientation = Orientation.Down if first.canvas_pos.y > second.canvas_pos.y else Orientation.Up
    origin = first.canvas_pos
    length = np.abs(second.canvas_pos.y - first.canvas_pos.y).astype(int)
    return Vector(orientation=orientation, length=length, origin=origin)


def get_point_for_match_shape_furthest(background_obj: Primitive, filter_obj: Primitive,
                                       match_shape_only: bool, try_unique: bool = True,
                                       padding: Surround = Surround(0, 0, 0, 0),
                                       transformations: List[str] = ('rotate', 'scale', 'flip', 'invert', 'colour')) -> Point:
    result = filter_obj.match_to_background(background_obj, match_shape_only=match_shape_only, try_unique=try_unique,
                                            padding=padding, transformations=transformations)
    match_positions = []
    for r in result:
        if r['rotate'] is None and r['scale'] is None:
            match_positions.append(r['translate_to_coordinates'])
    match_positions = match_positions[0]
    _, index = furthest_point_to_point(filter_obj.canvas_pos, match_positions)

    return match_positions[index]


def get_point_and_rotation_for_match_shape_furthest(background_obj: Primitive, filter_obj: Primitive,
                                                    match_shape_only: bool, try_unique: bool = True,
                                                    padding: Surround = Surround(0, 0, 0, 0)) -> Tuple[Point, int]:
    result = filter_obj.match_to_background(background_obj, match_shape_only=match_shape_only, try_unique=try_unique, padding=padding)
    match_positions = [result[i]['canvas_pos'][0] for i in range(len(result))]
    rotations = [result[i]['rotation'] for i in range(len(result))]
    _, index = furthest_point_to_point(filter_obj.canvas_pos, match_positions)

    return match_positions[index], rotations[index]


def get_point_for_match_shape_nearest(background_obj: Primitive, filter_obj: Primitive,
                                      match_shape_only: bool, try_unique:bool = True,
                                      padding: Surround = Surround(0, 0, 0, 0)) -> Point:
    result = filter_obj.match_to_background(background_obj, match_shape_only=match_shape_only, try_unique=try_unique, padding=padding)
    rotation = 0
    scale = 1
    match_positions = []
    for r in result:
        if r['rotation'] == rotation and r['scale'] == scale:
            match_positions.append(r['canvas_pos'])
    match_positions = match_positions[0]

    _, index = closest_point_to_point(filter_obj.canvas_pos, match_positions)

    return match_positions[index]


def get_point_and_rotation_for_match_shape_nearest(background_obj: Primitive, filter_obj: Primitive,
                                                   match_shape_only: bool, try_unique:bool = True,
                                                   padding: Surround = Surround(0, 0, 0, 0)) -> \
        tuple[int | list[Point], int | list[Point]]:
    result = filter_obj.match_to_background(background_obj, match_shape_only=match_shape_only, try_unique=try_unique, padding=padding)
    match_positions = [result[i]['canvas_pos'] for i in range(len(result))]
    rotations = [result[i]['rotation'] for i in range(len(result))]
    _, index = closest_point_to_point(filter_obj.canvas_pos, match_positions)

    return match_positions[index], rotations[index]


def get_point_and_rotation_for_best_match_to_objects(object_to_move: Primitive, target_objects: List[Primitive],
                                                     match_shape_only: bool = False) -> Tuple[int, Point]:
    best_match = 0
    best_rotate = 0
    best_translate_to_coordinates = None

    for to in target_objects:
        result = object_to_move.match_to_background(to, match_shape_only=match_shape_only, try_unique=True,
                                                    padding=Surround(0, 0, 0, 0), transformations=['rotate'])
        if result[0]['result'] > best_match:
            best_match = result[0]['result']
            best_rotate = result[0]['rotate']
            best_translate_to_coordinates = result[0]['translate_to_coordinates'][0]

    return best_rotate, best_translate_to_coordinates


def get_random_colours(not_included: List[Colour], number: int = 1, replace: bool = True) -> List[Colour] | Colour:
    return Colour.random(not_included=not_included, number=number, replace=replace)


def get_object_feature_colour_at_position(obj: Primitive, pos: Point) -> int:
    return int(obj.actual_pixels[int(pos.y - obj.canvas_pos.y), int(pos.x - obj.canvas_pos.x)])


def get_object_feature_colour(obj: Primitive) -> int:
    return obj.colour


def get_object_feature_all_colours(obj: Object) -> List[int]:
    """
    Get a list of all the colours of the obj
    :param obj: The Object to get the colours from -> Object
    :return: The list of colours -> List[int]
    """
    return list(obj.get_used_colours())


def get_object_feature_size(obj: Object) -> Dimension2D:
    """
    Returns the size of the object
    :param obj: The Object to get the size of -> Object
    :return: The size of the object -> Dimension2D
    """
    return obj.dimensions


def get_object_feature_size_x(obj: Object) -> int:
    """
    Returns the length (x) of the object
    :param obj: The Object to get the length of -> Object
    :return: The DImension2D.x of the object -> int
    """
    return obj.dimensions.dx


def get_object_feature_size_y(obj: Object) -> int:
    """
    Returns the height (y) of the Object
    :param obj: The Object to get the height of -> Object
    :return: The Dimension2D.y of the object -> int
    """
    return obj.dimensions.dy


def get_object_feature_canvas_pos(obj: Object) -> Point:
    """
    Gets the obj canvas position (the bottom left corner of the rectangle that defines the Object)
    :param obj: The Object -> Object
    :return: The canvas_pos Point -> Point
    """
    return obj.canvas_pos


def get_object_feature_canvas_pos_x(obj: Primitive) -> int:
    """
    Gets the obj x element (horizontal) of the canvas position (the bottom left corner of the rectangle that defines the Object)
    :param obj: The Object -> Object
    :return: The canvas_pos.x -> int
    """
    return obj.canvas_pos.x


def get_object_feature_canvas_pos_y(obj: Primitive) -> int:
    """
    Gets the obj y element (vertical) of the canvas position (the bottom left corner of the rectangle that defines the Object)
    :param obj: The Object -> Object
    :return: The canvas_pos.y -> int
    """
    return obj.canvas_pos.y


def get_object_feature_coloured_positions(obj: Primitive) -> List[Point]:
    pos = obj.get_coloured_pixels_positions()
    result = []
    for p in pos:
        result.append(Point(p[1], p[0]))

    return result


def get_object_feature_number_of_colours(obj: Primitive) -> int:
    return len(obj.get_coloured_pixels_positions())


def get_object_feature_relative_point_position(obj: Primitive, relative_point: RelativePoint) -> Point:
    return obj.relative_points[relative_point]


def get_object_feature_position_of_colour(obj: Primitive, colour: int) -> List[Point] | None:
    """
    Get the list of the positions (Points) of all the pixels of colour.
    :param obj: The Object to look for the coloured pixels.
    :param colour: The colour to look for.
    :return: A list of the positions (Points) of all the pixels of colour. Even if there is only one pixel this is a list. None is there are no pixels with that colour
    """
    positions = obj.get_coloured_pixels_positions(colour)
    if len(positions) == 0:
        return None
    elif len(positions) == 1:
        return [Point(x=positions[0][1], y=positions[0][0], z=obj.canvas_pos.z)]
    else:
        point_positions = []
        for p in positions:
            point_positions.append(Point(p[1], p[0], obj.canvas_pos.z))
        return point_positions


def get_object_feature_least_used_colour(obj: Primitive) -> int | List[int]:
    colours = list(obj.get_used_colours())
    amounts_of_colours = []
    for c in colours:
        amounts_of_colours.append(len(obj.get_coloured_pixels_positions(c)))

    least_colours = list(np.array(colours)[np.where(amounts_of_colours == np.min(amounts_of_colours))[0]])

    if len(least_colours) == 1:
        return least_colours[0]

    return least_colours


def get_object_feature_most_used_colour(obj: Primitive) -> int | List[int]:
    colours = list(obj.get_used_colours())
    amounts_of_colours = []
    for c in colours:
        amounts_of_colours.append(len(obj.get_coloured_pixels_positions(c)))

    most_colours = list(np.array(colours)[np.where(amounts_of_colours == np.max(amounts_of_colours))[0]])

    if len(most_colours) == 1:
        return most_colours[0]

    return most_colours


# Funcs to select Primitives
def select_all_objects(canvas: Canvas) -> List[Primitive]:
    return copy(canvas.objects)


def select_object_with_canvas_pos(canvas: Canvas, canvas_pos: Point) -> Primitive | None:
    for o in canvas.objects:
        if o.canvas_pos == canvas_pos:
            return copy(o)
    return None


def select_largest_object_by_area(canvas: Canvas) -> Primitive:
    new_canvas = copy(canvas)
    return new_canvas.sort_objects_by_size(used_dim='area')[-1]


def select_largest_object_by_height(canvas: Canvas) -> Primitive:
    new_canvas = copy(canvas)
    return new_canvas.sort_objects_by_size(used_dim='height')[-1]


def select_largest_object_by_width(canvas: Canvas) -> Primitive:
    new_canvas = copy(canvas)
    return new_canvas.sort_objects_by_size(used_dim='width')[-1]


def select_smallest_object_by_area(canvas: Canvas) -> Primitive:
    new_canvas = copy(canvas)
    return new_canvas.sort_objects_by_size(used_dim='area')[0]


def select_smallest_object_by_height(canvas: Canvas) -> Primitive:
    new_canvas = copy(canvas)
    return new_canvas.sort_objects_by_size(used_dim='height')[0]


def select_smallest_object_by_width(canvas: Canvas) -> Primitive:
    new_canvas = copy(canvas)
    return new_canvas.sort_objects_by_size(used_dim='width')[0]


def select_object_with_the_most_colours(canvas: Canvas) -> Primitive:
    objects = select_all_objects(canvas)
    n = 0
    good_object = None
    for o in objects:
        nn = get_object_feature_number_of_colours(o)
        if nn > n:
            n = nn
            good_object = copy(o)

    return good_object


def select_object_with_the_fewer_colours(canvas: Canvas) -> Primitive:
    objects = select_all_objects(canvas)
    n = 100
    good_object = None
    for o in objects:
        nn = get_object_feature_number_of_colours(o)
        if nn < n:
            n = nn
            good_object = copy(o)

    return good_object


def select_rest_of_the_objects(canvas: Canvas, obj: Primitive | List[Primitive] | None) -> List[Primitive]:
    temp_obj_list = [copy(o) for o in canvas.objects]
    if isinstance(obj, Primitive):
        temp_obj_list.remove(obj)
    if isinstance(obj, List):
        for o in obj:
            temp_obj_list.remove(o)

    return temp_obj_list


def select_all_objects_of_colour(canvas: Canvas, colour: int) -> List[Primitive]:
    new_canvas = copy(canvas)
    return new_canvas.get_objects_of_colour(colour)


def select_only_object_of_colour(canvas: Canvas, colour: int) -> Primitive | None:
    all_objects = select_all_objects_of_colour(canvas, colour=colour)
    if len(all_objects) > 0:
        return all_objects[0]
    else:
        return None


def select_objects_of_type(canvas: Canvas, primitive_type: type[Primitive]) -> List[Primitive]:
    new_canvas = copy(canvas)
    objs_of_type = []
    for obj in new_canvas.objects:
        if isinstance(obj, primitive_type):
            objs_of_type.append(copy(obj))

    return objs_of_type


def group_objects_according_to_colour(canvas: Canvas) -> Tuple[List[int], List[List[Primitive]]]:
    num_of_objects_per_group = []
    objects_in_a_group = []
    colours = get_canvas_feature_all_object_colours(canvas)
    for c in colours:
        objects_in_a_group.append(select_all_objects_of_colour(canvas, c))
        num_of_objects_per_group.append(len(objects_in_a_group[-1]))

    return num_of_objects_per_group, objects_in_a_group
# </editor-fold>


# <editor-fold desc="Functions to transform Primitives">
def object_transform_rotate(obj: Primitive, rotation: int) -> Primitive:
    new_obj = copy(obj)
    new_obj.rotate(times=rotation)
    return new_obj


def object_transform_translate_to_point(obj: Object, target_point: Point,
                                        object_point: Point | None = None) -> Object:
    """
    Translates an Object to a new position by moving the object_point Point to the target_point Point. If the
    object_point is None then the object_point becomes the canvas_pos Point of the obj.
    :param obj: The obj to translate -> Object
    :param target_point: The Point to translate the object_point to -> Point
    :param object_point: The Point in the Object (either the canvas_pos or another Point) to translate to the target_point -> Point
    :return: The translated Object -> Object
    """
    new_obj = copy(obj)
    new_obj.translate_to_coordinates(target_point=target_point, object_point=object_point)
    return new_obj


def object_transform_change_depth(obj: Primitive, target_depth: int) -> Primitive:
    new_obj = copy(obj)
    new_obj.translate_to_coordinates(target_point=Point(obj.canvas_pos.x,
                                                        obj.canvas_pos.y,
                                                        target_depth),
                                     object_point=obj.canvas_pos)
    return new_obj


def object_transform_translate_by_distance(obj: Primitive, distance: Dimension2D) -> Primitive:
    new_obj = copy(obj)
    new_obj.translate_by(distance=distance)
    return new_obj


def object_transform_translate_along_direction(obj: Primitive, direction: Vector) -> Primitive:
    new_obj = copy(obj)
    new_obj.translate_along(direction=direction)
    return new_obj


def object_transform_translate_relative_point_to_point(obj: Primitive, relative_point: RelativePoint,
                                                       other_point: Point) -> Primitive:
    new_obj = copy(obj)
    new_obj.translate_relative_point_to_point(relative_point=relative_point, other_point=other_point)
    return new_obj


def object_transform_translate_to_front_of_all(canvas: Canvas, obj: Primitive) -> Primitive:
    new_obj = copy(obj)
    max_z = 0
    for o in canvas.objects:
        if max_z < o.canvas_pos.z:
            max_z = o.canvas_pos.z

    new_obj.translate_to_coordinates(Point(new_obj.canvas_pos.x, new_obj.canvas_pos.y, max_z + 1))

    return new_obj


def object_transform_translate_to_back_of_all(canvas: Canvas, obj: Primitive) -> Primitive:
    new_obj = copy(obj)
    max_z = 1000
    for o in canvas.objects:
        if max_z < o.canvas_pos.z:
            max_z = o.canvas_pos.z

    new_obj.translate_to_coordinates(Point(new_obj.canvas_pos.x, new_obj.canvas_pos.y, max_z - 1))

    return new_obj


def object_transform_translate_to_front_of_object(obj: Primitive, other: Primitive) -> Primitive:
    new_obj = copy(obj)
    new_obj.translate_to_coordinates(Point(new_obj.canvas_pos.x, new_obj.canvas_pos.y, other.canvas_pos.z + 1))

    return new_obj


def object_transform_translate_to_back_of_object(obj: Primitive, other: Primitive) -> Primitive:
    new_obj = copy(obj)
    new_obj.translate_to_coordinates(Point(new_obj.canvas_pos.x, new_obj.canvas_pos.y, other.canvas_pos.z - 1))

    return new_obj


def object_transform_mirror(obj: Primitive, axis: Orientation):
    new_obj = copy(obj)
    new_obj.mirror(axis=axis, on_axis=False)
    return new_obj


def object_transform_mirror_on_axis(obj: Primitive, axis: Orientation):
    new_obj = copy(obj)
    new_obj.mirror(axis=axis, on_axis=True)
    return new_obj


def object_transform_flip_only(obj: Primitive, axis: Orientation | Vector):
    new_obj = copy(obj)
    if isinstance(axis, Vector):
        axis = axis.orientation
    new_obj.flip(axis=axis, translate=False)
    return new_obj


def object_transform_flip_and_translate(obj: Primitive, axis: Orientation | Vector):
    new_obj = copy(obj)
    if isinstance(axis, Vector):
        axis = axis.orientation
    new_obj.flip(axis=axis, translate=True)
    return new_obj


def object_transform_new_colour(obj: Primitive, colour: int) -> Object:
    """
    Makes all the non black pixels of an object that colour.
    :param obj: The Object to transform
    :param colour: The colour
    :return: The transformed Object
    """
    new_obj = copy(obj)
    colours = get_object_feature_all_colours(new_obj)
    for c in colours:
        if c != colour:
            new_obj.replace_colour(colour, c)
    return new_obj


def object_transform_negate(obj: Primitive) -> Primitive:
    new_obj = copy(obj)
    new_obj.negate_colour()
    return  new_obj


def object_transform_delete_colour(obj: Primitive, colour: int) -> Primitive:
    new_obj = copy(obj)
    new_obj.actual_pixels[np.where(new_obj.actual_pixels == colour)] = 1
    return new_obj


def object_transform_fill_holes(obj: Primitive, colour: int) -> Primitive:
    o = copy(obj)
    o.fill_holes(colour)
    return o


def object_transform_split_object_along_axis(obj: Primitive, cut_orientation: Orientation, percentage: float | None = None,
                                             pixels: int | None = None) -> Tuple[Primitive, Primitive]:
    ob_a, ob_b = obj.split_object_along_axis(cut_orientation=cut_orientation, percentage=percentage, pixels=pixels)
    predef_a = Predefined(actual_pixels=copy(ob_a.actual_pixels))
    predef_a.canvas_pos = ob_a.canvas_pos
    predef_b = Predefined(actual_pixels=copy(ob_b.actual_pixels))
    predef_b.canvas_pos = ob_b.canvas_pos

    return predef_a, predef_b


def object_transform_split_object_in_quarters(obj: Primitive, round_to_include: bool = True) -> Tuple[Primitive,
                                                                                                      Primitive,
                                                                                                      Primitive,
                                                                                                      Primitive]:

    ul_obj, ur_obj, dl_obj, dr_obj = obj.split_object_in_quarters(round_to_include=round_to_include)

    ul = Predefined(actual_pixels=ul_obj.actual_pixels)
    ul.canvas_pos = ul_obj.canvas_pos
    ur = Predefined(actual_pixels=ur_obj.actual_pixels)
    ur.canvas_pos = ur_obj.canvas_pos
    dl = Predefined(actual_pixels=dl_obj.actual_pixels)
    dl.canvas_pos = dl_obj.canvas_pos
    dr = Predefined(actual_pixels=dr_obj.actual_pixels)
    dr.canvas_pos = dr_obj.canvas_pos

    return ul, ur, dl, dr


def object_transform_split_object_by_colour(obj: Primitive) -> List[Primitive]:
    temp_canvas = Canvas(size=obj.dimensions)
    o = copy(obj)
    temp_canvas.add_new_object(o)
    temp_canvas.split_object_by_colour(o)
    objects = [ob for ob in temp_canvas.objects]
    return objects


def object_transform_add_two_objects(obj_a: Primitive, obj_b: Primitive) -> Primitive:
    return copy(obj_a + obj_b)
# </editor-fold>


# <editor-fold desc="Functions to order">
def order_over_indices(things: List[Any], order_indices:List[int], reverse: bool = False) -> List[Primitive]:
    """
    Gets a list of anything (Objects, Points, etc) and returns a list of the same things but ordered according to the order_indices.
    :param things: The list of things to order.
    :param reverse: If True then reverse the order.
    :return: The ordered list of the same things.
    """
    new_list = np.array(things)[order_indices]
    return [copy(o) for o in new_list]

def order_pixels_over_x(pixels: List[Point], reverse: bool = False) -> List[Point]:
    """
    Gets a list of Points and returns a list of the same Points but ordered according to their x position on the Canvas.
    :param objects: The list of Points to order.
    :param reverse: If True then reverse the order.
    :return: The order list of the same Points.
    """
    x_positions  = []
    for p in pixels:
        x_positions.append(p.x)
    indices = list(np.argsort(x_positions)) if not reverse else list(reversed(np.argsort(x_positions)))

    new_list = np.array(pixels)[indices]
    return [copy(p) for p in new_list]

def order_pixels_over_y(pixels: List[Point], reverse: bool = False) -> List[Point]:
    """
  Gets a list of Objects and returns a list of the same Points but ordered according to their y position on the Canvas.
  :param objects: The list of Points to order.
  :param reverse: If True then reverse the order.
  :return: The order list of the same Points.
  """
    y_positions  = []
    for p in pixels:
        y_positions.append(p.y)
    indices = list(np.argsort(y_positions)) if not reverse else list(reversed(np.argsort(y_positions)))

    new_list = np.array(pixels)[indices]
    return [copy(p) for p in new_list]

def order_pixels_over_z(pixels: List[Point], reverse: bool = False) -> List[Point]:
    """
  Gets a list of Objects and returns a list of the same Points but ordered according to their z (depth) position on the Canvas.
  :param objects: The list of Points to order.
  :param reverse: If True then reverse the order.
  :return: The order list of the same Points.
  """
    z_positions  = []
    for p in pixels:
        z_positions.append(p.y)
    indices = list(np.argsort(z_positions)) if not reverse else list(reversed(np.argsort(z_positions)))

    new_list = np.array(pixels)[indices]
    return [copy(p) for p in new_list]

def order_objects_over_height(objects: List[Primitive], reverse: bool = False) -> List[Primitive]:
    """
    Gets a list of Objects and returns a list of the same Objects but ordered according to their height.
    :param objects: The list of Objects to order.
    :param reverse: If True then reverse the order.
    :return: The order list of the same Objects.
    """
    heights = []
    for o in objects:
        heights.append(o.dimensions.dy)
    indices = list(np.argsort(heights)) if not reverse else list(reversed(np.argsort(heights)))
    new_list = np.array(objects)[indices]
    return [copy(o) for o in new_list]

def order_objects_over_x(objects: List[Primitive], reverse: bool = False) -> List[Primitive]:
    """
    Gets a list of Objects and returns a list of the same Objects but ordered according to their x position on the Canvas.
    :param objects: The list of Objects to order.
    :param reverse: If True then reverse the order.
    :return: The order list of the same Objects.
    """
    x_positions  = []
    for o in objects:
        x_positions.append(o.canvas_pos.x)
    indices = list(np.argsort(x_positions)) if not reverse else list(reversed(np.argsort(x_positions)))

    new_list = np.array(objects)[indices]
    return [copy(o) for o in new_list]

def order_objects_over_y(objects: List[Primitive], reverse: bool = False) -> List[Primitive]:
    """
  Gets a list of Objects and returns a list of the same Objects but ordered according to their y position on the Canvas.
  :param objects: The list of Objects to order.
  :param reverse: If True then reverse the order.
  :return: The order list of the same Objects.
  """
    y_positions  = []
    for o in objects:
        y_positions.append(o.canvas_pos.y)
    indices = list(np.argsort(y_positions)) if not reverse else list(reversed(np.argsort(y_positions)))

    new_list = np.array(objects)[indices]
    return [copy(o) for o in new_list]
# </editor-fold>


# <editor-fold desc="Functions to create Primitives">
def make_new_random(size: Dimension2D | np.ndarray | List, border_size: Surround = Surround(0, 0, 0, 0),
                    canvas_pos: Point = Point(0, 0), colour: None | int = None, occupancy_prob: float = 0.5,
                    required_dist_to_others: Surround = Surround(0, 0, 0, 0),
                    _id: None | int = None, actual_pixels_id: None | int = None, canvas_id: None | int = None) \
        -> Primitive:
    return Random(size=size, border_size=border_size, canvas_pos=canvas_pos, colour=colour,
                  occupancy_prob=occupancy_prob, required_dist_to_others=required_dist_to_others, _id=_id,
                  actual_pixels_id=actual_pixels_id, canvas_id=canvas_id)


def make_new_parallelogram(size: Dimension2D | np.ndarray | List, border_size: Surround = Surround(0, 0, 0, 0),
                           canvas_pos: Point = Point(0, 0), colour: None | int = None,
                           required_dist_to_others: Surround = Surround(0, 0, 0, 0),
                           _id: None | int = None, actual_pixels_id: None | int = None, canvas_id: None | int = None) \
        -> Primitive:
    return Parallelogram(size=size, border_size=border_size, canvas_pos=canvas_pos, colour=colour,
                         required_dist_to_others=required_dist_to_others, _id=_id,
                         actual_pixels_id=actual_pixels_id, canvas_id=canvas_id)


def make_new_cross(size: Dimension2D | np.ndarray | List, border_size: Surround = Surround(0, 0, 0, 0),
                   canvas_pos: Point = Point(0, 0), colour: None | int = None,
                   required_dist_to_others: Surround = Surround(0, 0, 0, 0),
                   _id: None | int = None, actual_pixels_id: None | int = None, canvas_id: None | int = None) \
        -> Primitive:
    return Cross(size=size, border_size=border_size, canvas_pos=canvas_pos, colour=colour,
                 required_dist_to_others=required_dist_to_others, _id=_id,
                 actual_pixels_id=actual_pixels_id, canvas_id=canvas_id)


def make_new_hole(size: Dimension2D | np.ndarray | List, thickness: Surround = Surround(1, 1, 1, 1),
                  border_size: Surround = Surround(0, 0, 0, 0), canvas_pos: Point = Point(0, 0),
                  colour: None | int = None, required_dist_to_others: Surround = Surround(0, 0, 0, 0),
                  _id: None | int = None, actual_pixels_id: None | int = None, canvas_id: None | int = None) \
        -> Primitive:
    return Hole(size=size, border_size=border_size, canvas_pos=canvas_pos, colour=colour, thickness=thickness,
                required_dist_to_others=required_dist_to_others, _id=_id,
                actual_pixels_id=actual_pixels_id, canvas_id=canvas_id)


def make_new_pi(size: Dimension2D | np.ndarray | List, border_size: Surround = Surround(0, 0, 0, 0),
                canvas_pos: Point = Point(0, 0), colour: None | int = None,
                required_dist_to_others: Surround = Surround(0, 0, 0, 0),
                _id: None | int = None, actual_pixels_id: None | int = None, canvas_id: None | int = None) \
        -> Primitive:
    return Pi(size=size, border_size=border_size, canvas_pos=canvas_pos, colour=colour,
              required_dist_to_others=required_dist_to_others, _id=_id,
              actual_pixels_id=actual_pixels_id, canvas_id=canvas_id)


def make_new_inverse_cross(height: int, border_size: Surround = Surround(0, 0, 0, 0),
                           canvas_pos: Point = Point(0, 0), colour: None | int = None,
                           required_dist_to_others: Surround = Surround(0, 0, 0, 0),
                           fill_colour: None | int = None, fill_height: None | int = None,
                           _id: None | int = None, actual_pixels_id: None | int = None, canvas_id: None | int = None) \
        -> Primitive:
    return InverseCross(height=height, border_size=border_size, canvas_pos=canvas_pos, colour=colour,
                        required_dist_to_others=required_dist_to_others, _id=_id,
                        fill_colour=fill_colour, fill_height=fill_height,
                        actual_pixels_id=actual_pixels_id, canvas_id=canvas_id)


def make_new_dot(border_size: Surround = Surround(0, 0, 0, 0),
                 canvas_pos: Point = Point(0, 0), colour: None | int = None,
                 required_dist_to_others: Surround = Surround(0, 0, 0, 0),
                 _id: None | int = None, actual_pixels_id: None | int = None, canvas_id: None | int = None) \
        -> Primitive:
    return Dot(border_size=border_size, canvas_pos=canvas_pos, colour=colour,
               required_dist_to_others=required_dist_to_others, _id=_id,
               actual_pixels_id=actual_pixels_id, canvas_id=canvas_id)


def make_new_angle(size: Dimension2D | np.ndarray | List, border_size: Surround = Surround(0, 0, 0, 0),
                   canvas_pos: Point = Point(0, 0), colour: None | int = None,
                   required_dist_to_others: Surround = Surround(0, 0, 0, 0),
                   _id: None | int = None, actual_pixels_id: None | int = None, canvas_id: None | int = None) \
        -> Primitive:
    return Angle(size=size, border_size=border_size, canvas_pos=canvas_pos, colour=colour,
                 required_dist_to_others=required_dist_to_others, _id=_id,
                 actual_pixels_id=actual_pixels_id, canvas_id=canvas_id)


def make_new_diagonal(height: int, border_size: Surround = Surround(0, 0, 0, 0),
                      canvas_pos: Point = Point(0, 0), colour: None | int = None,
                      required_dist_to_others: Surround = Surround(0, 0, 0, 0),
                      _id: None | int = None, actual_pixels_id: None | int = None, canvas_id: None | int = None) \
        -> Primitive:
    return Diagonal(height=height, border_size=border_size, canvas_pos=canvas_pos, colour=colour,
                    required_dist_to_others=required_dist_to_others, _id=_id,
                    actual_pixels_id=actual_pixels_id, canvas_id=canvas_id)


def make_new_steps(height: int, depth: int, border_size: Surround = Surround(0, 0, 0, 0),
                   canvas_pos: Point = Point(0, 0), colour: None | int = None,
                   required_dist_to_others: Surround = Surround(0, 0, 0, 0),
                   _id: None | int = None, actual_pixels_id: None | int = None, canvas_id: None | int = None) \
        -> Primitive:
    return Steps(height=height, depth=depth, border_size=border_size, canvas_pos=canvas_pos, colour=colour,
                 required_dist_to_others=required_dist_to_others, _id=_id,
                 actual_pixels_id=actual_pixels_id, canvas_id=canvas_id)


def make_new_fish(border_size: Surround = Surround(0, 0, 0, 0),
                  canvas_pos: Point = Point(0, 0), colour: None | int = None,
                  required_dist_to_others: Surround = Surround(0, 0, 0, 0),
                  _id: None | int = None, actual_pixels_id: None | int = None, canvas_id: None | int = None) \
        -> Primitive:
    return Fish(border_size=border_size, canvas_pos=canvas_pos, colour=colour,
                required_dist_to_others=required_dist_to_others, _id=_id,
                actual_pixels_id=actual_pixels_id, canvas_id=canvas_id)


def make_new_bolt(_center_on: bool = False, border_size: Surround = Surround(0, 0, 0, 0),
                  canvas_pos: Point = Point(0, 0), colour: None | int = None,
                  required_dist_to_others: Surround = Surround(0, 0, 0, 0),
                  _id: None | int = None, actual_pixels_id: None | int = None, canvas_id: None | int = None) \
        -> Primitive:
    return Bolt(_center_on=_center_on, border_size=border_size, canvas_pos=canvas_pos, colour=colour,
                required_dist_to_others=required_dist_to_others, _id=_id,
                actual_pixels_id=actual_pixels_id, canvas_id=canvas_id)


def make_new_tie(border_size: Surround = Surround(0, 0, 0, 0),
                 canvas_pos: Point = Point(0, 0), colour: None | int = None,
                 required_dist_to_others: Surround = Surround(0, 0, 0, 0),
                 _id: None | int = None, actual_pixels_id: None | int = None, canvas_id: None | int = None) \
        -> Primitive:
    return Tie(border_size=border_size, canvas_pos=canvas_pos, colour=colour,
               required_dist_to_others=required_dist_to_others, _id=_id,
               actual_pixels_id=actual_pixels_id, canvas_id=canvas_id)


def make_new_spiral(size: Dimension2D | np.ndarray | List, border_size: Surround = Surround(0, 0, 0, 0),
                    canvas_pos: Point = Point(0, 0), colour: None | int = None, gap: int = 1,
                    required_dist_to_others: Surround = Surround(0, 0, 0, 0),
                    _id: None | int = None, actual_pixels_id: None | int = None, canvas_id: None | int = None) \
        -> Primitive:
    return Spiral(size=size, border_size=border_size, canvas_pos=canvas_pos, colour=colour,
                  required_dist_to_others=required_dist_to_others, _id=_id, gap=gap,
                  actual_pixels_id=actual_pixels_id, canvas_id=canvas_id)


def make_new_pyramid(height: int, border_size: Surround = Surround(0, 0, 0, 0),
                     canvas_pos: Point = Point(0, 0), colour: None | int = None,
                     required_dist_to_others: Surround = Surround(0, 0, 0, 0),
                     _id: None | int = None, actual_pixels_id: None | int = None, canvas_id: None | int = None) \
        -> Primitive:
    return Pyramid(height=height, border_size=border_size, canvas_pos=canvas_pos, colour=colour,
                   required_dist_to_others=required_dist_to_others, _id=_id,
                   actual_pixels_id=actual_pixels_id, canvas_id=canvas_id)


def make_new_maze(size: Dimension2D | np.ndarray | List, border_size: Surround = Surround(0, 0, 0, 0),
                  colour: None | int = None, required_dist_to_others: Surround = Surround(0, 0, 0, 0),
                  _id: None | int = None, actual_pixels_id: None | int = None, canvas_id: None | int = None) \
        -> Primitive:
    return Maze(size=size, border_size=border_size, colour=colour, required_dist_to_others=required_dist_to_others,
                _id=_id, actual_pixels_id=actual_pixels_id, canvas_id=canvas_id)
# </editor-fold>
