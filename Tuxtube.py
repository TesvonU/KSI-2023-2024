import numpy as np
import numpy.typing as npt
import cv2 as cv
from typing import Tuple
import math

Frame_t = npt.NDArray[np.uint8]
Pixel_t = npt.NDArray[np.uint8]


class VideoEditor():
    def __init__(self) -> None:
        self.videos = []
        self.cuts = []
        self.edits_stack = []
        self.add_subway = False

    def add_video(self, path: str) -> 'VideoEditor':
        self.videos.append(path)
        return self

    def grayscale(self, start: float, end: float) -> 'VideoEditor':
        self.edits_stack.append(["gray", start, end])
        return self

    def epilepsy(self, start: float, end: float) -> 'VideoEditor':
        self.edits_stack.append(["epilepsy", start, end])
        return self

    def chromakey(self, start: float, end: float, img: str,
                  color: Tuple[int, int, int],
                  similarity: int) -> 'VideoEditor':
        self.edits_stack.append(["chroma", start, end, img, color, similarity])
        return self

    def cut(self, start: float, end: float) -> 'VideoEditor':
        self.cuts.append([start, end])
        return self

    def shaky_cam(self, start: float, end: float) -> 'VideoEditor':
        self.edits_stack.append(["shake", start, end])
        return self

    def subway(self, path):
        self.add_subway = path
        return self

    def image(self, start: float, end: float, img: str,
              pos: Tuple[float, float, float, float]) -> 'VideoEditor':
        self.edits_stack.append(["img", start, end, img, pos])
        return self

    def short_passed(self, current, previous, shorting):
        '''spočítá rozdíl mezi 2 obrázky (předchozí obrázek je poslední
         použitý frame) potom seřte hodnotu rozdílů, vydělí 765,
        vypočítá průměrnou hodnotu a porovná s thresholdem pro podobnost
        '''
        if not shorting:
            return True
        if previous is None:
            return True
        current = np.array(current)
        previous = np.array(previous)
        difference = abs(current - previous)
        total_difference = np.sum(difference, axis=-1)
        devided = total_difference / 765
        avarage = ((np.sum(devided) / np.size(devided)) * 100)
        if avarage < 10:
            return False
        else:
            return True

    def create_grayscale(self, start: float, end: float, frame):
        '''převede začátek a konec na framy, pokud je frame v intervalu,
         udělá z něj šedý, převod tam a zpět zachová 3 kanály'''
        if start == math.inf:
            start = self.current_lenght
        if end == math.inf:
            end = self.current_lenght
        start = round(start * self.framerate)
        end = round(end * self.framerate)
        if start <= self.frame_index <= end:
            frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            frame = cv.cvtColor(frame, cv.COLOR_GRAY2BGR)
        return frame

    def create_img(self, start: float, end: float, img: str,
                   pos: Tuple[float, float, float, float], frame):
        '''převede začátek a konec na framy, pokud je frame v intervalu,
           vybere zónu kam se má obrázek umístit, změní velikost přidávaného
           obrázku na velikost zóny, pokud má obrázek alfa kanál, tak se
            zpracuje - https://stackoverflow.com/questions/40895785/
            using-opencv-to-overlay-transparent-image-onto-another-image,
             pokud ne, tak se jen vymění pixely'''
        if start == math.inf:
            start = self.current_lenght
        if end == math.inf:
            end = self.current_lenght
        start = round(start * self.framerate)
        end = round(end * self.framerate)
        if start <= self.frame_index <= end:
            rows = frame.shape[0]
            columns = frame.shape[1]
            zone = frame[round(rows * pos[1]):round(rows * pos[3]),
                    round(columns * pos[0]):round(columns * pos[2])]
            image = cv.imread(img, -1)
            rows2 = zone.shape[0]
            columns2 = zone.shape[1]
            image = cv.resize(image, (columns2, rows2))
            if image.shape[2] == 4:
                alpha = image[:, :, 3] / 255
                for k in range(0, 3):
                    zone[:, :, k] = (1 - alpha) * zone[:, :,
                                                  k] + alpha * image[:, :, k]
            else:
                zone = image
            frame[round(rows * pos[1]):round(rows * pos[3]),
            round(columns * pos[0]):round(columns * pos[2])] = zone
        return frame

    def create_chroma(self, start: float, end: float, img: str,
                      color: Tuple[int, int, int],
                      similarity: int, frame):
        '''převede začátek a konec na framy, pokud je frame v intervalu zvětší
        obrázek, který přidáváme, porovná rozdíly barva/pixel, pokud je jejich
        součet větší než similiarity jsou v nové matici označeny jako True,
        pomocí téhle matice vybereme oblast, kterou nahradíme'''
        if start == math.inf:
            start = self.current_lenght
        if end == math.inf:
            end = self.current_lenght
        start = round(start * self.framerate)
        end = round(end * self.framerate)
        if start <= self.frame_index <= end:
            color = (color[2], color[1], color[0])
            image = cv.imread(img)
            image = cv.resize(image, (frame.shape[1], frame.shape[0]))
            difference = abs(frame - color)
            total_difference = np.sum(difference, axis=-1)
            new_pixles = total_difference < similarity
            frame[new_pixles] = image[new_pixles]
        return frame

    def create_shake(self, start: float, end: float, frame):
        '''převede začátek a konec na framy, pokud je frame v intervalu,
         udělá z něj šedý, převod tam a zpět zachová 3 kanály'''
        if start == math.inf:
            start = self.current_lenght
        if end == math.inf:
            end = self.current_lenght
        start = round(start * self.framerate)
        end = round(end * self.framerate)
        if start <= self.frame_index <= end:
            state = (self.frame_index % 4)
            rows = frame.shape[0]
            columns = frame.shape[1]
            if state == 0:
                frame2 = frame[0:rows - 15, 0:columns - 15]
            if state == 3:
                frame2 = frame[0:rows - 15, 15:columns]
            if state == 2:
                frame2 = frame[15:rows, 15:columns]
            if state == 1:
                frame2 = frame[15:rows, 0:columns - 15]
            frame = cv.resize(frame2, (columns, rows))
        return frame

    def create_epilepsy(self, start: float, end: float, frame):
        '''převede začátek a konec na framy, pokud je frame v intervalu, podle
        čísla snímku určí jakou barvu nakombinuje s obrázkem
         '''
        if start == math.inf:
            start = self.current_lenght
        if end == math.inf:
            end = self.current_lenght
        start = round(start * self.framerate)
        end = round(end * self.framerate)
        if start <= self.frame_index <= end:
            color = np.zeros(frame.shape, dtype=np.uint8)
            state = (self.frame_index % 7)
            if state == 0:
                color[:, :] = [0, 0, 255]
            if state == 1:
                color[:, :] = [0, 127, 255]
            if state == 2:
                color[:, :] = [0, 255, 255]
            if state == 3:
                color[:, :] = [0, 255, 0]
            if state == 4:
                color[:, :] = [255, 0, 0]
            if state == 5:
                color[:, :] = [130, 0, 70]
            if state == 6:
                color[:, :] = [211, 0, 148]
            frame = cv.addWeighted(frame, 0.7, color, 0.3, 0)
            # print(state)
        return frame

    def create_subway(self, frame, sub_frame):
        '''
        funkce určí oblast kam dá Subway video, resizne původní video tak, ať
        se vleze do ponechané oblasti a spojí frame s původním videem a subwaye
        '''
        rows = frame.shape[0]
        columns = frame.shape[1]
        columns_left = round(columns / 4)
        # cv.imshow("sub", sub_frame)
        # cv.waitKey(0)
        sub_frame = cv.resize(sub_frame, (columns_left, rows))
        frame2 = cv.resize(frame, (columns - columns_left, rows))
        frame[:, 0:columns_left] = sub_frame
        frame[:, columns_left:] = frame2
        return frame

    def render(self, path: str, width: int, height: int, framerate: float,
               short: bool = False) -> 'VideoEditor':
        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        out = cv.VideoWriter(path, fourcc, framerate, (width, height), True)
        self.frame_index = 0
        real_frame = 0
        subway_true = False
        if self.add_subway:
            answers = input(
                "Using subway will change video framerate to 25fps,"
                " if you use different source video with different framerate,"
                " please input it's framerate,"
                " do you want to input custom framerate? Yes/No ")
            framerate = 25
            if answers == "Yes":
                fps_input = int(
                    input("Enter framerate as integer (example: 30): "))
                framerate = fps_input
                if type(fps_input) != int or framerate < 0 or framerate > 300:
                    print("Invalid input, 25 will be used instead")
                    framerate = 25
            subway_true = True
            sub_video = cv.VideoCapture(self.add_subway)

        # Sběr/výpočet informací pro přidání/odebrání snímkůu u změny FPS
        for current_video in self.videos:
            self.framerate = framerate
            video = cv.VideoCapture(current_video)
            current_width = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
            current_height = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))
            self.current_fps = video.get(cv.CAP_PROP_FPS)
            current_frames = int(video.get(cv.CAP_PROP_FRAME_COUNT))
            self.current_lenght = current_frames / self.current_fps
            wanted_frames = int(self.current_lenght * framerate)
            matching_size = width == current_width and height == current_height
            matching_lenght = wanted_frames == current_frames
            neccesary_frames = wanted_frames - current_frames
            need_more = wanted_frames >= current_frames
            matching_fps = self.current_fps == framerate
            counter = 1
            if neccesary_frames != 0:
                if not need_more:
                    n = abs(current_frames / neccesary_frames)
                if need_more:
                    n = abs(neccesary_frames / current_frames)
            else:
                n = 0.99
            # print(matching_size, matching_fps)
            # print(n)
            # print("want:", wanted_frames,"curr", current_frames,
            # "neccesarry:", neccesary_frames)
            # print("need more:", need_more, "matching lenght:",
            # matching_lenght)
            last_frame = None
            n2 = n

            # výpočet oblasti pro Cut - převod sekund na framy
            cuts_frames = []
            for cut in self.cuts:
                if cut[0] == math.inf:
                    cut[0] = self.current_lenght
                if cut[1] == math.inf:
                    cut[1] = self.current_lenght
                cuts_frames.append([round(cut[0] * self.framerate),
                                    round(cut[1] * self.framerate)])
            while True:
                ret, frame = video.read()
                if not ret:
                    print("Nepodařilo se načíst snímek (Konec videa?).")
                    break
                real_frame += 1
                # print(self.frame_index)
                '''kontrola, jestli máme snímek odebrat, kvůli short=True nebo
                 Cut, Cut proběhne místo možného zkrácení videa'''
                cutting = False
                for cut in cuts_frames:
                    if cut[0] <= self.frame_index <= cut[1]:
                        cutting = True
                        print("will cut", self.frame_index, real_frame)
                if cutting:
                    self.frame_index += 1
                short_checked = self.short_passed(frame, last_frame, short)
                if short_checked and not cutting:
                    last_frame = frame.copy()

                # oprava velikosti framu
                if not matching_size:
                    frame = cv.resize(frame, (width, height))

                # úpravy videa jsou ve jednom seznamu a volají se postupně
                for edit in self.edits_stack:
                    if edit[0] == "gray":
                        frame = self.create_grayscale(edit[1], edit[2], frame)
                    if edit[0] == "img":
                        frame = self.create_img(edit[1], edit[2], edit[3],
                                                edit[4], frame)
                    if edit[0] == "chroma":
                        frame = self.create_chroma(edit[1], edit[2], edit[3],
                                                   edit[4], edit[5], frame)
                    if edit[0] == "shake":
                        frame = self.create_shake(edit[1], edit[2], frame)
                    if edit[0] == "epilepsy":
                        frame = self.create_epilepsy(edit[1], edit[2], frame)

                '''pokud je počet FPS stejný snímek si vždy přidá, pokud je
                 třeba více snímků, přidá se za každý snímek n snímků (zbytek
                  z n se přidá později), pokud je třeba odebrat snímek, odebere
                  se každý ntý n je pro tyto dva případy jiná hodnota, pokud
                  se má přidat subway, přidá se 1 znímek ke každému přidanému
                   snímku'''
                if not need_more:
                    if counter < n2:
                        if short_checked and not cutting:
                            if subway_true:
                                ret2, sub_frame = sub_video.read()
                                if ret2:
                                    frame = self.create_subway(frame,
                                                               sub_frame)
                                else:
                                    print("Subway footage ended")
                                    sub_video = cv.VideoCapture(
                                        self.add_subway)
                            out.write(frame)
                            self.frame_index += 1
                            cv.imshow('frame', frame)
                            # print("low")
                    else:
                        n2 += n
                        if cutting:
                            self.frame_index -= 1

                elif matching_lenght or need_more:
                    if short_checked and not cutting:
                        if subway_true:
                            ret2, sub_frame = sub_video.read()
                            if ret2:
                                frame = self.create_subway(frame, sub_frame)
                            else:
                                print("Subway footage ended")
                                sub_video = cv.VideoCapture(self.add_subway)
                        out.write(frame)
                        self.frame_index += 1
                        cv.imshow('frame', frame)
                        # print("casual")
                    while n2 >= 1:
                        if short_checked and not cutting:
                            if subway_true:
                                ret2, sub_frame = sub_video.read()
                                if ret2:
                                    frame = self.create_subway(frame,
                                                               sub_frame)
                                else:
                                    print("Subway footage ended")
                                    sub_video = cv.VideoCapture(
                                        self.add_subway)
                            self.frame_index += 1
                            cv.imshow('frame', frame)
                            # print("high")
                        n2 -= 1
                    if not matching_lenght:
                        n2 += n
                counter += 1
                if cv.waitKey(25) == ord('q'):
                    break

            video.release()

        out.release()
        cv.destroyAllWindows()
        return self


if __name__ == "__main__":
    VideoEditor().add_video("clean.mp4").epilepsy(0, 6).render("diffvideo.mp4", 426, 240, 25, True)
