from enum import Enum, auto
from typing import Callable, List
from pathlib import Path
import numpy as np
import cv2 as cv



class Location(Enum):
    CITY = auto()
    NATURE = auto()


class PartOfDay(Enum):
    SUNRISESUNSET = auto()
    DAY = auto()
    NIGHT = auto()


class Season(Enum):
    SPRINGSUMMER = auto()
    AUTUMN = auto()
    WINTER = auto()


class Notes(Enum):
    NOTES = auto()
    NOT_NOTES = auto()


def classify_location(image: np.ndarray) -> Location:
    '''funkce změní kanály na HSV, jelikož se z nich lépe vybírá zelená
    pokud pixel spadá do range green1 - green2, vybere se jako zelený, pak se
    vypočítá procento zelených pixelů, pak se obrázek změní na grayscale a vybe
    rou se světlejší pixely, které bývají na fotkách měst - cesty atd. a vypo
    čítá se jejich procento na obrázku, pak se tyto dvě procenta od sebe
    odečtou, pokud je výsledek větší než threshhold, je obrázek příroda'''
    img: np.ndarray = image
    size: int = img.shape[0] * img.shape[1]
    img_hsv: np.ndarray = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    img_gray: np.ndarray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    green1: np.ndarray = np.array([30, 40, 40])
    green2: np.ndarray = np.array([90, 255, 255])
    # HSV barvy jsou v poměru 0.5, 2.55, 2.55 s https://colorizer.org/
    green_detection: np.ndarray = cv.inRange(img_hsv, green1, green2)
    # cv2.imshow('green_detection', green_detection)
    # cv2.waitKey(0)
    green_detected: tuple = np.where(green_detection == 255)
    green_total: int = len(green_detected[0])
    green_percent: float = (green_total / size) * 100
    gray_detection: np.ndarray
    _: np.ndarray
    _, gray_detection = cv.threshold(img_gray, 200, 255, cv.THRESH_BINARY)
    # cv.imshow("gray",gray_detection)
    # cv.waitKey(0)
    gray_detected: tuple = np.where(gray_detection == 255)
    gray_total: int = len(gray_detected[0])
    gray_percent: float = (gray_total / size) * 100
    total: float = green_percent - gray_percent
    # print(total)
    if total >= 20:
        return Location.NATURE
    return Location.CITY


def classify_time_of_day(image: np.ndarray) -> PartOfDay:
    '''funkce změní kanály na HSV, jelikož se z nich lépe vybírá oranžová
    pokud pixel spadá do range orange1 - orange2, vybere se jako oranžový,
    a spočítá se jejich % na obrázku, toto by měl být odstín, který vytvoří
    západ/východ slunce, pak se obrázek změní na grayscale a určí se průměrný
    odstín, což určí jak světlý/tmavý obrázek je, určí se thresholdy šedé,
    podle něj se rozdělí den/noc/západ a východ, pro západ a východ platí,
    že na obrázku musí být alespoň trochu oranžového odstínu, jinak bude
    obrázek vyhodnocen jako den/noc, podle toho, k čemu má blíže'''
    img: np.ndarray = image
    size: int = img.shape[0] * img.shape[1]
    img_hsv: np.ndarray = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    img_gray: np.ndarray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    orange1: np.ndarray = np.array([0, 100, 100])
    orange2: np.ndarray = np.array([35, 255, 255])
    orange_detection: np.ndarray = cv.inRange(img_hsv, orange1, orange2)
    orange_detected: tuple = np.where(orange_detection == 255)
    orange_total: int = len(orange_detected[0])
    orange_percent: float = (orange_total / size) * 100
    gray_average: float = np.sum(img_gray) / img_gray.size
    gray_day: int = 120
    gray_night: int = 40
    if gray_average > gray_day:
        return PartOfDay.DAY
    if gray_average < gray_night:
        return PartOfDay.NIGHT

    if gray_average > 110:
        if orange_percent < 2:
            return PartOfDay.DAY
    if gray_average < 50:
        if orange_percent < 2:
            return PartOfDay.NIGHT
    if orange_percent < 0.1:
        if gray_average > 80:
            return PartOfDay.DAY
    return PartOfDay.SUNRISESUNSET


def classify_notes(image: np.ndarray) -> Notes:
    '''funkce vybere pixely, která mají kanály s hodnotami většími než 110
    a s hodnotami, které od sebe nesjou moc daleko od sebe. Pixely, které
    tohle splňují by měli být pixely bílého/našedlého pozadí,
    na kterém může být nějaký stín. (nepodařilo se mi najít vhodný range v HSV)
    Vypočítá se jejich % na obrázku, pokud je jich víc než 60%,vezmou se
    ostatní pixely, převedou se z matice s boolean hodnotami na matici
    s černo-bílým obrázkem as v něm se najdou kontury, ty,
    které mají area ne až moc malý, ale ne až moc velký by mohly být písmena,
    pokud je potenciálních písmen alespoň 60, na obrázku by mohl být text, tedy
    by to mohly být poznámky'''
    img: np.ndarray = image
    size: int = img.shape[0] * img.shape[1]
    condition1: np.ndarray = np.all(img > 110, axis=-1)
    difference: np.ndarray = np.max(img, axis=-1) - np.min(img, axis=-1)
    condition2: np.ndarray = difference < 26
    white_detected: np.ndarray = np.logical_and(condition1, condition2)
    white_total: int = np.sum(white_detected)
    white_percent: float = (white_total / size) * 100
    if 60 < white_percent < 99.5:
        other_pixels: np.ndarray = np.logical_not(white_detected)
        other_pixels = other_pixels.astype(np.uint8)
        contours: List[np.ndarray]
        _: np.ndarray
        contours, _ = cv.findContours(other_pixels, cv.RETR_TREE, 2)
        letters: int = 0
        for contour in contours:
            area: float = cv.contourArea(contour)
            if 3 < area < 200:
                letters += 1
        if letters > 50:
            return Notes.NOTES
    return Notes.NOT_NOTES


def classify_season(image: np.ndarray) -> Season:
    '''
    funkce první rozdělí obrázek napůl, aby při detekci nebyl brán v potaz
    horizont, který není tolik ovlivněn obdobím
    vybere pixely, která mají kanály s hodnotami většími než 140
    a s hodnotami, které od sebe nesjou moc daleko od sebe. Pixely, které
    tohle splňují by měli být pixely sněhu. Vypočítá se jejich % na obrázku,
    funkce změní kanály na HSV, jelikož se z nich lépe vybírá modrá
    pokud pixel spadá do range blue1 - blue2, vybere se jako modrý,
    a spočítá se jejich % na obrázku, modrá reprezentuje chladné barvy typické
    pro zimu, toto procento se vydělí dvěmi - je méně významné než sníh,
    obě procenta se sečtou a pokud jsou větší než threshold, je obrázek zima,
    pokud ne, stejným způsobem, tentokrát na celém obrázku, se vybere
    zelená - stromy, zeleň, žlutá - pole, oranžová - podzimní listy, porovnají
    se jejich % na obrázku a podle něj se vybere, jestli je obrázek podzim, či
    jaro a léto
    '''
    img: np.ndarray = image
    img_half: np.ndarray = img[round(image.shape[0] / 2):]
    img_hsv: np.ndarray = cv.cvtColor(img_half, cv.COLOR_BGR2HSV)
    size: int = img_half.shape[0] * img_half.shape[1]
    condition1: np.ndarray = np.all(img_half > 140, axis=-1)
    difference: np.ndarray = np.max(img_half, axis=-1) - np.min(img_half, axis=-1)
    condition2: np.ndarray = difference < 30
    white_detected: np.ndarray = np.logical_and(condition1, condition2)
    white_total: int = np.sum(white_detected)
    white_percent: float = (white_total / size) * 100
    blue1: np.ndarray = np.array([85, 35, 40])
    blue2: np.ndarray = np.array([125, 255, 255])
    blue_detection: np.ndarray = cv.inRange(img_hsv, blue1, blue2)
    blue_detected: tuple = np.where(blue_detection == 255)
    blue_total: int = len(blue_detected[0])
    blue_percent: float = (blue_total / size) * 100
    total: float = white_percent + (blue_percent / 2)
    if total > 30:
        return Season.WINTER
        # print("total:", round(total, 2), passed)
    # print("total:", round(total, 2), passed)
    size = img.shape[0] * img.shape[1]
    img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    green1: np.ndarray = np.array([30, 40, 40])
    green2: np.ndarray = np.array([75, 255, 255])
    green_detection: np.ndarray = cv.inRange(img_hsv, green1, green2)
    green_detected: tuple = np.where(green_detection == 255)
    green_total: int = len(green_detected[0])
    green_percent: float = (green_total / size) * 100
    orange1: np.ndarray = np.array([0, 70, 50])
    orange2: np.ndarray = np.array([20, 255, 255])
    orange_detection: np.ndarray = cv.inRange(img_hsv, orange1, orange2)
    orange_detected: tuple = np.where(orange_detection == 255)
    orange_total: int = len(orange_detected[0])
    orange_percent: float = (orange_total / size) * 100
    yellow1: np.ndarray = np.array([20, 60, 60])
    yellow2: np.ndarray = np.array([40, 255, 255])
    yellow_detection: np.ndarray = cv.inRange(img_hsv, yellow1, yellow2)
    yellow_detected: tuple = np.where(yellow_detection == 255)
    yellow_total: int = len(yellow_detected[0])
    yellow_percent: float = (yellow_total / size) * 100

    # les, kde jsou některé listy oranžové
    if green_percent > 15 and orange_percent > 5 and yellow_percent < 30:
        return Season.AUTUMN

    # pole
    if green_percent < 1:
        if (green_percent + yellow_percent) > orange_percent:
            return Season.SPRINGSUMMER
        return Season.AUTUMN

    # ostatní
    if ((green_percent + yellow_percent) / 2) > (orange_percent * 1.3):
        return Season.SPRINGSUMMER
    return Season.AUTUMN

# ----------------------------------------------------
# --- Below this line is just code that runs tests ---
# ----------------------------------------------------

