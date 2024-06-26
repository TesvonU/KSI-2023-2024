from typing import Optional

def play(board_size: int, pos: str) -> Optional[list[str]]:
    if board_size in [2, 3, 4, 6, 8]:
        return None
    if board_size == 1:
        return []
    letters = ["a", "b", "c", "d", "e", "f", "g", "h", "i"]
    numbers = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]
    for loop in range(9 - board_size):
        letters.pop()
        numbers.pop()
    try:
        pos = [board_size - int(pos[3]), letters.index(pos[2])]
    except:
        return None
    result = []
    moveable = [pos]
    unmovable = []
    new_pos1 = None
    new_pos2 = None
    previously_movable = [pos]
    returned = []
    backtrack(returned, board_size, result, new_pos1, new_pos2, moveable, unmovable, previously_movable)
    if returned:
        return returned[0:len(returned)]
    else:
        return None


def backtrack(returned, board_size, result, new_pos1, new_pos2, movable, unmovable, previously_movable):
    currently_movable = [x for x in movable if x not in unmovable]

    if len(movable) == board_size ** 2:
        returned.extend(result)
        return True

    for horse in currently_movable:
        pos = horse
        move_x = [2, 1, -1, -2, -2, -1, 1, 2]
        move_y = [1, 2, 2, 1, -1, -2, -2, -1]

        for k in range(8):
            new_pos1 = [pos[0] + move_x[k], pos[1] + move_y[k]]
            if is_valid2(board_size, new_pos1):
                for i in range(8):
                    new_pos2 = [pos[0] + move_x[i], pos[1] + move_y[i]]
                    if is_valid2(board_size, new_pos2):
                        if is_valid(new_pos1, new_pos2, movable):
                            print(result)
                            if backtrack(returned, board_size, result + [(f"{chr(pos[1] + 97)}{board_size - pos[0]}qK{chr(new_pos1[1] + 97)}{board_size - new_pos1[0]}&{chr(new_pos2[1] + 97)}{board_size - new_pos2[0]}")], new_pos1, new_pos2, movable + [new_pos1, new_pos2], unmovable + [pos], movable): #inspirace z https://stackoverflow.com/questions/18544419/how-to-convert-numbers-to-alphabet
                                return True
    return False


def is_valid(new_pos1, new_pos2, previously_movable):
    if new_pos1:
        if new_pos1 in previously_movable or new_pos2 in previously_movable:
            return False
        if new_pos1 == new_pos2:
            return False
    return True


def is_valid2(board_size, new_pos):
    if new_pos:
        if new_pos[0] < 0 or new_pos[1] < 0:
            #print("nula false")
            return False
        if new_pos[0] >= board_size or new_pos[1] >= board_size:
            #print("strop false")
            return False
    #print("accept")
    return True

#Test
play(5, "qKc4")
