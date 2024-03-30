from typing import Dict, List, Union

# TypovĂ© konstanty
Piece = List[List[int]]
Piece_Dict = Dict[str, Piece]
Bag = List[List[str]]


class Objekt:
    def __init__(self, color: str, pieces: Piece_Dict):
        self.color = str(color)
        self.positions = pieces[self.color]

    def get_new_cord(self, central_cord: List[int]):
        first_position = self.positions[0]
        up_x = central_cord[0] - first_position[0]
        up_y = central_cord[1] - first_position[1]
        return [
            [cord[0] + up_x, cord[1] + up_y]
            for cord in self.positions
        ]


def solve(height: int, width: int, pieces: Piece_Dict) -> Bag:
    stack = []
    for piece in pieces:
        stack.append(Objekt(piece, pieces))
    bag = []
    mapable = []
    for x in range(width):
        for y in range(height):
            bag.append([x, y])
    tested_item_index = 0
    position = "start"
    result = []
    item_index = 0
    picked_up = []
    win = len(stack)

    backtrack(win, item_index, picked_up, stack, bag, tested_item_index,
              height, width, position, result, mapable)
    if result:
        return result
    else:
        return None


def backtrack(win: int, item_index: int, picked_up: List[List[int]],
              stack: List[Objekt], bag: Bag, tested_item_index: int,
              height: int, width: int, position: List[int],
              result: List[List[int]], mapable: List[Piece_Dict]):
    if showmap(mapable, width, height)[0]:
        result.extend(showmap(mapable, width, height)[1])
        return True
    temp_bag = [x for x in bag if x not in picked_up]
    temp_stack = stack[tested_item_index]
    if position != "start":
        stack.remove(stack[tested_item_index])
    for position in temp_bag:
        if is_valid(stack, temp_bag, tested_item_index, position):
            if backtrack(win, item_index + 1, picked_up + [x for x in
                                                           stack[
                                            tested_item_index].get_new_cord(
                                                               position)],
                         stack, bag,
                         tested_item_index, height, width, position, result,
                         mapable +
                         [{stack[tested_item_index].color: [x for x in
                                                            stack[
                                            tested_item_index].get_new_cord(
                                                                position)]}]):
                return True
    if temp_stack not in stack:
        stack.insert(0, temp_stack)
    return False


def showmap(mapable: List[Piece_Dict], width: int, height: int):
    new_mapable = {}
    for item in mapable:
        new_mapable.update(item)
    rows, cols = (height, width)
    map = [["-"] * cols] * rows
    map = [list(x) for x in map]
    a = 0
    for key in new_mapable:
        for position in new_mapable[key]:
            map[position[1]][position[0]] = key
            a += 1
    if a == width * height:
        return [True, map]
    else:
        return [False, None]


def is_valid(stack: List[Objekt], bag: Bag,
             tested_item_index: int, position: List[int], ):
    if position == "start":
        return True
    new_pos = stack[tested_item_index].get_new_cord(position)
    return all(element in bag for element in new_pos)



def count_solutions(height: int, width: int, pieces: Piece_Dict) -> int:
    # TODO
    raise NotImplementedError


def solve_non_rect(bag: Bag, pieces: Piece_Dict) -> Bag:
    # TODO
    raise NotImplementedError



# -------------------------- PRINT FUNCTIONS -------------------------- #


def bag_print(bag: Bag, color: bool = True) -> None:
    """
    Tuto funkci mĹŻĹľete pouĹľĂ­vat pro grafickĂ© zobrazenĂ­ batohu

    UmĂ­ vypisovat barevnÄ›, ale nĂ© kaĹľdĂ˝ terminĂˇl to umĂ­ zobrazit.
    BarevnĂ© zobrazenĂ­ rozhodnÄ› funguje v terminĂˇlu Visual Studio Code.
    Pokud VĂˇĹˇ terminĂˇl neumĂ­ zobrazovat 24bitovĂ© ANSI sekvence, mĹŻĹľete
    barevnost vypnout nastavenĂ­m parametru `color` na `False`
    """
    colors = {"A": "\u001b[48;2;255;215;180m",  # Apricot
              "B": "\u001b[48;2;0;0;255m",  # Blue
              "C": "\u001b[48;2;70;240;240m",  # Cyan
              "D": "\u001b[48;2;245;130;48m",  # Orange
              "E": "\u001b[48;2;240;50;230m",  # Magenta
              "F": "\u001b[48;2;128;128;128m",  # Gray
              "G": "\u001b[48;2;60;180;75m",  # Green
              "H": "\u001b[48;2;170;110;40m",  # Brown
              "I": "\u001b[48;2;250;190;212m",  # Pink
              "J": "\u001b[48;2;255;250;200m",  # Beige
              "K": "\u001b[48;2;0;0;0m",  # Black
              "L": "\u001b[48;2;210;245;60m",  # Lime
              "M": "\u001b[48;2;128;0;0m",  # Maroon
              "N": "\u001b[48;2;0;0;128m",  # Navy
              "O": "\u001b[48;2;128;128;0m",  # Olive
              "P": "\u001b[48;2;145;30;180m",  # Purple
              "Q": "\u001b[48;2;170;255;195m",  # Mint
              "R": "\u001b[48;2;255;0;0m",  # Red
              "S": "\u001b[48;2;220;190;255m",  # Levander
              "T": "\u001b[48;2;0;128;128m",  # Teal
              "U": "\u001b[48;2;116;10;255m",  # Violet
              "V": "\u001b[48;2;224;255;102m",  # Uranium
              "W": "\u001b[48;2;255;255;255m",  # White
              "X": "\u001b[48;2;148;255;181m",  # Jade
              "Y": "\u001b[48;2;255;255;25m",  # Yellow
              "Z": "\u001b[48;2;255;80;5m"  # Zinnia
              }
    end = "\u001b[97m\u001b[40m"
    sb = []
    for y in bag:
        for x in y:
            if x == "":
                x = " "
            if len(x) != 1:
                x = x[0:1]
            if color:
                sb.append(f"{colors[x] if x in colors else end} {x} {end}")
            else:
                sb.append(f" {x} ")
        sb.append("\n")


def print_piece(piece: Piece) -> None:
    """
    graficke zobrazeni predmetu
    """
    minx = min(piece, key=lambda p: p[0])[0]
    miny = min(piece, key=lambda p: p[1])[1]
    maxx = max(piece, key=lambda p: p[0])[0]
    maxy = max(piece, key=lambda p: p[1])[1]
    sb = []
    sb.append("   ")
    for x in range(minx, maxx + 1):
        sb.append(f"{' ' if x >= 0 else ''}{str(x)}")
    sb.append("\n")
    for y in range(miny, maxy + 1):
        sb.append(f"{' ' if y >= 0 else ''}{str(y)} ")
        for x in range(minx, maxx + 1):
            if [x, y] in piece:
                sb.append("â–“â–“")
            else:
                sb.append("â–‘â–‘")
        sb.append("\n")
    print("".join(sb))


if __name__ == "__main__":
    # testy

    p = {
        "B": [[0, -1], [0, 0], [2, 1], [1, 1], [1, 0]],
        "G": [[0, 0], [1, 0], [0, -1], [0, -2]],
        "R": [[0, -1], [1, -1], [1, 0]],
        "Y": [[0, 2], [0, 0], [0, 1]]
    }

    # solve(3, 5, p)

    p = {
        "I": [[0, 0], [0, 1], [0, 2], [0, 3], [0, 4]],
        "T": [[0, 0], [0, 1], [0, 2], [-1, 2], [1, 2]],
        "P": [[0, 0], [0, 1], [0, 2], [1, 1], [1, 2]],
        "X": [[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1]],
        "W": [[0, 0], [1, 0], [1, 1], [2, 1], [2, 2]],
        "Y": [[0, 0], [1, 0], [2, 0], [2, 1], [3, 0]],
        "F": [[0, 0], [1, 0], [1, -1], [1, 1], [2, 1]],
        "N": [[0, 0], [0, 1], [1, 1], [1, 2], [1, 3]],
        "L": [[0, 0], [0, 1], [1, 0], [0, 2], [0, 3]],
        "Z": [[0, 0], [1, 0], [1, -1], [1, -2], [2, -2]],
        "V": [[0, 0], [1, 0], [2, 0], [2, 1], [2, 2]],
        "U": [[0, 0], [0, 1], [1, 0], [2, 0], [2, 1]],
    }

    # bag_print(solve(6, 10, p))
