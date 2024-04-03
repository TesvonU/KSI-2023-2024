import pyautogui
import cv2 as cv
import numpy as np
from typing import Tuple, List

#bohužel jsem neměl moc času na to fixnout zbývající bugy - způsobená tím, že
#kódy byly psány na dvou verzích opencv, kod mi ale lokálně funguje
# bohuzel neni prehledny, typovany
#a komentovany (je 23:55 :/)

def get_hint(minefield: List[str]) -> List[str]:
    field = []
    for row in minefield:
        line = []
        for zone in row:
            line.append(zone)
        field.append(line)
    field = np.array(field)
    answer = np.zeros(field.shape)
    field2 = np.zeros((field.shape[0]+2, field.shape[1]+2), dtype=str)
    field2[:,:] = "x"
    field2[1:-1,1:-1] = field
    if minefield[0][0] == "0":
        print("thids")
    #print("test")
    numbers = np.array(["1", "2", "3", "4", "5", "6", "7", "8"])
    #kontorlovane pole
    copy_field2 = field2.copy()
    for row_x in range(copy_field2[1:-1].shape[0]):
        row_x += 1
        for zone_y in range(copy_field2[row_x,1:-1].shape[0]):
            zone_y +=1
            if copy_field2[row_x, zone_y] in numbers:
                checked_area = copy_field2[row_x-1:row_x+2, zone_y-1:zone_y+2] #3x3 matice
                #print(checked_area)
                M_detection = np.where(checked_area == "M")
                M_detected = len(M_detection[0])
                if M_detected > int(copy_field2[row_x, zone_y]):
                    for pos_x in range(checked_area.shape[0]):
                        for pos_y in range(checked_area.shape[1]) :
                            if checked_area[pos_x, pos_y] == "M":
                                field2[row_x + (-1 + pos_x), zone_y + (-1 + pos_y)] = "?"
                                answer[row_x - 1+(-1 + pos_x), zone_y - 1 + (-1 + pos_y)] = 1
      #first green check
    for row_x in range(field2[1:-1].shape[0]):
        row_x += 1
        for zone_y in range(field2[row_x, 1:-1].shape[0]):
            zone_y += 1
            if field2[row_x, zone_y] in numbers:
                checked_area = field2[row_x - 1:row_x + 2,
                               zone_y - 1:zone_y + 2]  # 3x3 matice
                # print(checked_area)
                M_detection = np.where(checked_area == "M")
                M_detected = len(M_detection[0])  # + nebezpecna pole
                cislo = int(field2[row_x, zone_y])
                if cislo == M_detected:
                    for pos_x in range(checked_area.shape[0]):
                        for pos_y in range(checked_area.shape[1]):
                            if checked_area[pos_x, pos_y] == "?":
                                    if answer[row_x - 1 + (-1 + pos_x), zone_y - 1 + (-1 + pos_y)] != 1:
                                        answer[row_x - 1 + (-1 + pos_x), zone_y - 1 + (-1 + pos_y)] = 2
    #first red check
    for row_x in range(field2[1:-1].shape[0]):
        row_x += 1
        for zone_y in range(field2[row_x,1:-1].shape[0]):
            zone_y +=1
            if field2[row_x, zone_y] in numbers:
                checked_area = field2[row_x-1:row_x+2, zone_y-1:zone_y+2] #3x3 matice
                #print(checked_area)
                M_detection = np.where(checked_area == "M")
                M_detected = len(M_detection[0])
                cislo_minus_vlajka = int(field2[row_x, zone_y]) - M_detected
                free_detection = np.where(checked_area == "?")
                free_detected = len(free_detection[0])
                if cislo_minus_vlajka == free_detected:
                    for pos_x in range(checked_area.shape[0]):
                        for pos_y in range(checked_area.shape[1]):
                            if checked_area[pos_x, pos_y] == "?":
                                if answer[row_x - 1+(-1 + pos_x), zone_y - 1 + (-1 + pos_y)] == 1:
                                    answer[row_x - 1 + (-1 + pos_x), zone_y - 1 + (-1 + pos_y)] = 4
                                elif answer[row_x - 1+(-1 + pos_x), zone_y - 1 + (-1 + pos_y)] == 0:
                                    answer[row_x - 1+(-1 + pos_x), zone_y - 1 + (-1 + pos_y)] = 3
    #second green check # will be repeated
    for row_x in range(field2[1:-1].shape[0]):
        row_x += 1
        for zone_y in range(field2[row_x,1:-1].shape[0]):
            zone_y +=1
            if field2[row_x, zone_y] in numbers:
                checked_area = field2[row_x-1:row_x+2, zone_y-1:zone_y+2] #3x3 matice
                #print(checked_area)
                M_detection = np.where(checked_area == "M")
                M_detected = len(M_detection[0])    # + nebezpecna pole
                for pos_x in range(checked_area.shape[0]):
                    for pos_y in range(checked_area.shape[1]):
                        if checked_area[pos_x, pos_y] == "?":
                            if answer[row_x - 1 + (-1 + pos_x), zone_y - 1 + (-1 + pos_y)] == 3 or answer[row_x - 1 + (-1 + pos_x), zone_y - 1 + (-1 + pos_y)] == 4:
                                M_detected += 1
                cislo = int(field2[row_x, zone_y])
                if cislo == M_detected:
                    for pos_x in range(checked_area.shape[0]):
                        for pos_y in range(checked_area.shape[1]):
                            if checked_area[pos_x, pos_y] == "?":
                                if answer[row_x - 1 + (-1 + pos_x), zone_y - 1 + (-1 + pos_y)] == 0:
                                    answer[row_x - 1+(-1 + pos_x), zone_y - 1 + (-1 + pos_y)] = 2
    #second red check will be repaeted?
    for row_x in range(field2[1:-1].shape[0]):
        row_x += 1
        for zone_y in range(field2[row_x,1:-1].shape[0]):
            zone_y +=1
            if field2[row_x, zone_y] in numbers:
                checked_area = field2[row_x-1:row_x+2, zone_y-1:zone_y+2] #3x3 matice
                #print(checked_area)
                M_detection = np.where(checked_area == "M")
                M_detected = len(M_detection[0])
                cislo_minus_vlajka = int(field2[row_x, zone_y]) - M_detected
                free_detection = np.where(checked_area == "?") # - bezpecna
                free_detected = len(free_detection[0])
                for pos_x in range(checked_area.shape[0]):
                    for pos_y in range(checked_area.shape[1]):
                        if checked_area[pos_x, pos_y] == "?":
                            if answer[row_x - 1 + (-1 + pos_x), zone_y - 1 + (-1 + pos_y)] == 2:
                                free_detected += -1
                if cislo_minus_vlajka == free_detected:
                    for pos_x in range(checked_area.shape[0]):
                        for pos_y in range(checked_area.shape[1]):
                            if checked_area[pos_x, pos_y] == "?":
                                if answer[row_x - 1+(-1 + pos_x), zone_y - 1 + (-1 + pos_y)] == 1:
                                    answer[row_x - 1 + (-1 + pos_x), zone_y - 1 + (-1 + pos_y)] = 4
                                elif answer[row_x - 1+(-1 + pos_x), zone_y - 1 + (-1 + pos_y)] == 0:
                                    answer[row_x - 1+(-1 + pos_x), zone_y - 1 + (-1 + pos_y)] = 3
    for row_x in range(field2[1:-1].shape[0]):
        row_x += 1
        for zone_y in range(field2[row_x,1:-1].shape[0]):
            zone_y +=1
            if field2[row_x, zone_y] in numbers:
                checked_area = field2[row_x-1:row_x+2, zone_y-1:zone_y+2] #3x3 matice
                #print(checked_area)
                M_detection = np.where(checked_area == "M")
                M_detected = len(M_detection[0])    # + nebezpecna pole
                for pos_x in range(checked_area.shape[0]):
                    for pos_y in range(checked_area.shape[1]):
                        if checked_area[pos_x, pos_y] == "?":
                            if answer[row_x - 1 + (-1 + pos_x), zone_y - 1 + (-1 + pos_y)] == 3 or answer[row_x - 1 + (-1 + pos_x), zone_y - 1 + (-1 + pos_y)] == 4:
                                M_detected += 1
                cislo = int(field2[row_x, zone_y])
                if cislo == M_detected:
                    for pos_x in range(checked_area.shape[0]):
                        for pos_y in range(checked_area.shape[1]):
                            if checked_area[pos_x, pos_y] == "?":
                                if answer[row_x - 1 + (-1 + pos_x), zone_y - 1 + (-1 + pos_y)] == 0:
                                    answer[row_x - 1+(-1 + pos_x), zone_y - 1 + (-1 + pos_y)] = 2
    #second red check will be repaeted?
    for row_x in range(field2[1:-1].shape[0]):
        row_x += 1
        for zone_y in range(field2[row_x,1:-1].shape[0]):
            zone_y +=1
            if field2[row_x, zone_y] in numbers:
                checked_area = field2[row_x-1:row_x+2, zone_y-1:zone_y+2] #3x3 matice
                #print(checked_area)
                M_detection = np.where(checked_area == "M")
                M_detected = len(M_detection[0])
                cislo_minus_vlajka = int(field2[row_x, zone_y]) - M_detected
                free_detection = np.where(checked_area == "?") # - bezpecna
                free_detected = len(free_detection[0])
                for pos_x in range(checked_area.shape[0]):
                    for pos_y in range(checked_area.shape[1]):
                        if checked_area[pos_x, pos_y] == "?":
                            if answer[row_x - 1 + (-1 + pos_x), zone_y - 1 + (-1 + pos_y)] == 2:
                                free_detected += -1
                if cislo_minus_vlajka == free_detected:
                    for pos_x in range(checked_area.shape[0]):
                        for pos_y in range(checked_area.shape[1]):
                            if checked_area[pos_x, pos_y] == "?":
                                if answer[row_x - 1+(-1 + pos_x), zone_y - 1 + (-1 + pos_y)] == 1:
                                    answer[row_x - 1 + (-1 + pos_x), zone_y - 1 + (-1 + pos_y)] = 4
                                elif answer[row_x - 1+(-1 + pos_x), zone_y - 1 + (-1 + pos_y)] == 0:
                                    answer[row_x - 1+(-1 + pos_x), zone_y - 1 + (-1 + pos_y)] = 3
    odpoved = []
    for line in answer:
        part = ""
        for character in line:
            part += str(int(character))
        odpoved.append(part)
    return odpoved


def recolor_3(zone, allowed_pixels): #neloopuje, hleda vechno najednou asi urcite ryhclejsi?
    #print(zone[0, 0, np.newaxis])
    #print(print(zone[0, 0].shape))
    #print(zone[0, 0, np.newaxis].shape)
    abs_distance = np.linalg.norm(allowed_pixels - zone[:, :, np.newaxis], axis=3) #zdroj https://stackoverflow.com/questions/29241056/how-do-i-use-np-newaxis
    best_pixel = np.argmin(abs_distance, axis=2)
    # print(zone[row, column], "na:", allowed_pixels[best_pixel])
    zone = allowed_pixels[best_pixel]
    return zone


def gamestate(image, minefield, grid: Tuple[np.ndarray, np.ndarray]) -> List[str]:
    left_x = minefield[0][0]
    right_x = minefield[1][0]
    top_y = minefield[0][1]
    bottom_y = minefield[1][1]
    img = image[top_y:bottom_y, left_x:right_x]
    squares = []
    img2 = img.copy()
    allowed_pixels = np.array(
        [[198, 198, 198], [128, 128, 128], [255, 255, 255], [255, 0, 0], [0, 128, 0], [0, 0, 255], [128, 0, 0], [0, 0, 128], [128, 128, 0], [0, 0, 0]])
    for y in range(len(grid[1]) - 1):
        for x in range(len(grid[0]) - 1):
            squares.append([grid[0][x] - left_x, grid[0][x+1] - left_x, grid[1][y] - top_y, grid[1][y+1] - top_y])

    odpoved = ""
    for k in range(len(squares)):  #range(len(squares))
        counts = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        names = ["light_gray:", "dark_gray:", "white:", "blue:", "green:", "red:", "dark_blue:", "brown:", "cyan:", "black:"]
        zone = img2[squares[k][2]:squares[k][3], squares[k][0]:squares[k][1]] #prvni zavorka je temporary
        size = zone.shape[0] * zone.shape[1]
        zone = recolor_3(zone, allowed_pixels)
        for k in range(zone.shape[0]):
            for i in range(zone.shape[1]):
                current_pixel = zone[k, i]
                counter = 0
                for color in allowed_pixels:
                    if np.all(color == current_pixel):
                        counts[counter] = counts[counter] + 1
                        break
                    counter += 1
        counts = (counts / size) * 100
        '''
        for xd in range(10):
            if counts[xd] > 0:
                print(names[xd], str(counts[xd]) + "%", end=" ")
        print(" ")
        '''
        if counts[names.index("blue:")] > 12:
            odpoved += "1"
        elif counts[names.index("green:")] > 15:
            odpoved += "2"
        elif counts[names.index("red:")] > 15:
            odpoved += "3"
        elif counts[names.index("dark_blue:")] > 13:
            odpoved += "4"
        elif counts[names.index("brown:")] > 15:
            odpoved += "5"
        elif counts[names.index("cyan:")] > 15:
            odpoved += "6"
        elif counts[names.index("red:")] > 2 and counts[names.index("black:")] > 5 and counts[names.index("dark_gray:")] > 7 and counts[names.index("brown:")] > 0.5:
            odpoved += "M"
        elif counts[names.index("black:")] > 10:
            odpoved += "7"
            print("7")
            for xd in range(10):
                if counts[xd] > 0:
                    print(names[xd], str(counts[xd]) + "%", end=" ")
            print(" ")
        elif counts[names.index("light_gray:")] > 80:
            odpoved += "0"
        elif counts[names.index("light_gray:")] > 50 and counts[names.index("dark_gray:")] > 5 and counts[names.index("white:")] > 5:
            odpoved += "?"
        elif counts[names.index("dark_gray:")] > 15:
            odpoved += "8"
            print("8")
            for xd in range(10):
                if counts[xd] > 0:
                    print(names[xd], str(counts[xd]) + "%", end=" ")
            print(" ")
        else:
            print("#error:")
            for xd in range(10):
                if counts[xd] > 0:
                    print(names[xd], str(counts[xd]) + "%", end=" ")
            print(" ")
    odpoved = [odpoved[i:i + (len(grid[0]) - 1)] for i in range(0, len(odpoved), (len(grid[0]) - 1))]
    return odpoved


def get_grid(image, minefield):
    left_x = minefield[0][0]
    right_x = minefield[1][0]
    top_y = minefield[0][1]
    bottom_y = minefield[1][1]
    centered = image[top_y:bottom_y, left_x:right_x]
    centered2 = centered.copy()
    allowed_pixels = np.array([[198, 198, 198], [128, 128, 128], [255, 255, 255]])
    selected = np.logical_not(np.isin(centered, allowed_pixels).all(axis=-1))
    centered2[selected] = [198, 198, 198]
    allowed_pixels = np.array([255, 255, 255])
    selected = np.isin(centered, allowed_pixels).all(axis=-1)
    centered2[selected] = [128, 128, 128]
    image_gray = cv.cvtColor(centered2, cv.COLOR_BGR2GRAY)
    _, thresh = cv.threshold(image_gray, 150, 255, cv.THRESH_BINARY_INV)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL,  cv.CHAIN_APPROX_SIMPLE)
    max_area = 0
    for contour in contours:
        if cv.contourArea(contour) > max_area:
            max_area = cv.contourArea(contour)
            max_contour = contour
    x_new, y_new, width, height = cv.boundingRect(max_contour)
    real_centered = centered[y_new:y_new + height, x_new:x_new + width]
    distance_x = left_x + x_new
    distance_y = top_y + y_new
    allowed_pixels = np.array([[198, 198, 198], [128, 128, 128], [255, 255, 255]])
    selected = np.logical_not(np.isin(real_centered, allowed_pixels).all(axis=-1))
    real_centered[selected] = [198, 198, 198]
    #print(minefield)
    #print("lines:", len(real_centered))
    currently_light = False
    x = []
    k = 0
    for column in real_centered[10]:
        if not currently_light:
            if np.all(column == [198,198,198]):
                x.append(k)
                currently_light = True
        else:
            if np.all(column != [198,198,198]):
                currently_light = False
        k += 1
    x.append(width - x[0])
    #x.append(x[-1] + (x[-1] - x[-2])-3)
    for i in range(len(x)):
        x[i] = x[i] + distance_x
    currently_light = False
    y = []
    k = 0
    for line in real_centered:
        if not currently_light:
            if np.all(line[10] == [198, 198, 198]):
                y.append(k)
                currently_light = True
        else:
            if np.all(line[10] != [198, 198, 198]):
                currently_light = False
        k += 1
    #y.append(y[-1] + (y[-1] - y[-2]) - 3)
    y.append(height - y[0])
    for i in range(len(y)):
        y[i] = y[i] + distance_y
    final_x = np.array(x)
    final_y = np.array(y)
    return (final_x, final_y)



monitor_x, monitor_y = pyautogui.size()
left_side = (0, 0, monitor_x // 2, monitor_y)
cv.namedWindow('Right Window')
cv.moveWindow('Right Window', monitor_x // 2, 0)
while True:
    left = pyautogui.screenshot(region=left_side)
    left = np.array(left)
    left = cv.cvtColor(left, cv.COLOR_RGB2BGR)
    gray_img = cv.cvtColor(left, cv.COLOR_BGR2GRAY)
    _, binary = cv.threshold(gray_img, 200, 255,cv.THRESH_BINARY)
    gray_negative = cv.bitwise_not(binary)
    edging = cv.Canny(gray_negative, 100, 200)
    contours, hierarchy = cv.findContours(edging, cv.RETR_TREE, 2)
    hierarchy = hierarchy[0]

    '''z kontur aproximujeme polynomy a určíme, že mají 4 strany,
     pak vybereme ten s největším obsahem a vrátíme jeho souřadnice
    '''

    rectangles = []
    for cnt in contours:
        rectangle_odchylka = cv.approxPolyDP(cnt,
                             0.05 * cv.arcLength(cnt, True), True)
        if len(rectangle_odchylka) == 4:
            rectangles.append(cnt)
    max_area = 0
    largest_area = ""
    for cnt in rectangles:
        if cv.contourArea(cnt) > max_area:
            max_area = cv.contourArea(cnt)
            largest_area = cnt
    try:
        highlighted = cv.drawContours(left.copy(), [largest_area],
                                      -1, (0, 255, 0), 3)
        x, y, width, height = cv.boundingRect(largest_area)
        x += 5
        y += 5
        width -= 5
        height -= 5
        #print(cv.boundingRect(nejvetsi_contura))

        field_pos = ((x-4, y), ((x + width, y + height)))
    except:
        pass
    try:
        grid = get_grid(left, field_pos)
        #for k in grid[1]:
        #    left[k, :] = [0, 0, 255]
        #for k in grid[0]:
            #left[:, k] = [0, 0, 255]
        left = pyautogui.screenshot(region=left_side)
        left = np.array(left)
        left = cv.cvtColor(left, cv.COLOR_RGB2BGR)
        central = left[y:y + height, x - 4:x + width]
        game_state = gamestate(left, field_pos, grid)
        hints = get_hint(game_state)
    except:
        pass
    try:
        top_y3 = -1
        bot_y3 = 0
        left = cv.cvtColor(left, cv.COLOR_BGR2BGRA)
        left[:, :, 3] = 255
        view = left.copy()
        for k in hints:
            top_y3 +=1
            bot_y3 +=1
            left_x3 = -1
            right_x3 = 0
            for number in k:
                left_x3 += 1
                right_x3 += 1
                to_color_with = number
                if number == "1":
                    yellow = np.zeros((grid[1][bot_y3]-grid[1][top_y3], grid[0][right_x3]-grid[0][left_x3], 4), dtype=np.uint8)
                    yellow[:, :] = (0, 255, 255, 128) #ma alfa kanal
                    view[grid[1][top_y3]:grid[1][bot_y3], grid[0][left_x3]:grid[0][right_x3], :] = cv.addWeighted(left[grid[1][top_y3]:grid[1][bot_y3], grid[0][left_x3]:grid[0][right_x3], :], 1, yellow, 0.4, 0)
                if number == "2":
                    yellow = np.zeros((grid[1][bot_y3]-grid[1][top_y3], grid[0][right_x3]-grid[0][left_x3], 4), dtype=np.uint8)
                    yellow[:, :] = (0, 255, 0, 128) #ma alfa kanal
                    view[grid[1][top_y3]:grid[1][bot_y3],
                    grid[0][left_x3]:grid[0][right_x3], :] = cv.addWeighted(
                        left[grid[1][top_y3]:grid[1][bot_y3],
                        grid[0][left_x3]:grid[0][right_x3], :], 1, yellow, 0.4, 0)
                if number == "3":
                    yellow = np.zeros((grid[1][bot_y3]-grid[1][top_y3], grid[0][right_x3]-grid[0][left_x3], 4), dtype=np.uint8)
                    yellow[:, :] = (0, 0, 255, 128) #ma alfa kanal
                    view[grid[1][top_y3]:grid[1][bot_y3],
                    grid[0][left_x3]:grid[0][right_x3], :] = cv.addWeighted(
                        left[grid[1][top_y3]:grid[1][bot_y3],
                        grid[0][left_x3]:grid[0][right_x3], :], 1, yellow, 0.4, 0)
                if number == "4":
                    yellow = np.zeros((grid[1][bot_y3]-grid[1][top_y3], grid[0][right_x3]-grid[0][left_x3], 4), dtype=np.uint8)
                    yellow[:, :] = (255, 0, 255, 128) #ma alfa kanal
                    view[grid[1][top_y3]:grid[1][bot_y3], grid[0][left_x3]:grid[0][right_x3], :] = cv.addWeighted(left[grid[1][top_y3]:grid[1][bot_y3], grid[0][left_x3]:grid[0][right_x3], :], 1, yellow, 0.4, 0)
    except:
        pass
    central = view[y:y + height, x - 4:x + width]
    cv.imshow('Right Window', central)
    key_pressed = cv.waitKey(1) #čeká 1ms
    if key_pressed != -1:
        break
