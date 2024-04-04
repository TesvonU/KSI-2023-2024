import math
from typing import Tuple, List
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import json
import numpy as np

SIZE_X = SIZE_Y = 100


class Asteroid:
    def __init__(self, x, y, mass, name, direction, speed, typ, acceleration):
        self.x = x
        self.y = y
        self.mass = mass
        self.name = name
        self.direction = direction
        self.speed = speed
        self.acceleration = acceleration
        self.xspeed = 0
        self.yspeed = 0
        self.exploded = False
        self.explosion_frame = 0
        self.explosion_pos = None


class Rocket:
    def __init__(self):
        self.x = 50
        self.y = 50
        self.og_mass = 3500 + 10  # palivo před jedním zrychlením
        self.mass = 3500  # palivo po jednom zrychlení
        self.direction = 43 + 90 + 180
        """
        raketa zrychlí jen jednou a to o 0.01 na tuto rychlost, poté buded
        odpálena gravitací, kdyby bylo první zrychlení menší, kvůli úspoře
        paliva, tak by byl efekt gravitace o hodně větší, jelikož by raketa
        v prvním snímku byla blíž zemi. Jelikož zrychlí raketa jen jednou,
        revnou víme, kolik spotřebujue paliva -> 10kq
        """
        self.speed = 0.01
        self.acceleration = 0
        self.xspeed = 0
        self.yspeed = 0
        self.move_frame = 0
        self.exploded = False
        self.fuel = 0


def calculate_gravitational_force(x: float, y: float, mass: float,
                                  target_x: float, target_y: float) \
        -> Tuple[float, float]:
    """
    Calculates the gravitational force between two points
    :param x: center of X point
    :param y: center of Y point
    :param mass: mass of the point
    :param target_x: influenced point's X
    :param target_y: influenced point's Y
    :return: difference of X-speed and Y-speed based on
    """
    gravity_strength = (mass * 10 ** -28) ** 0.5
    distance_x = target_x - x
    distance_y = target_y - y
    distance = (distance_x ** 2 + distance_y ** 2) ** 0.5
    if distance == 0:
        return 0, 0
    force_x = (distance_x / distance) * gravity_strength / distance
    force_y = (distance_y / distance) * gravity_strength / distance
    return force_x, force_y


def calculate_gravity_affected_speed(x: float, y: float, mass: float,
                                     target_x: float, target_y: float,
                                     target_speed_x: float,
                                     target_speed_y: float) \
        -> Tuple[float, float]:
    """
    Calculates new X-speed and Y-speed of a target point
    """
    force_x, force_y = calculate_gravitational_force(x, y, mass, target_x,
                                                     target_y)
    return target_speed_x - force_x, target_speed_y - force_y


def direction_and_speed_to_x_speed_and_y_speed(direction: float,
                                               speed: float) \
        -> Tuple[float, float]:
    """
    Converts direction and speed into a 2D vector representing the x speed
    and y speed.
    :param direction: direction in degrees (0-360)
    :param speed: speed in km/h
    :return: x-speed and y-speed in km/h
    """
    x_speed = speed * math.cos(math.radians(direction))
    y_speed = speed * math.sin(math.radians(direction))
    return x_speed, y_speed


def xy_speed_to_direction_and_speed(x_speed: float, y_speed: float) \
        -> Tuple[float, float]:
    # pomocí pythagorovi věty a funkce vypočítá směr a rychlost
    speed = math.sqrt(x_speed ** 2 + y_speed ** 2)
    direction = math.degrees(math.atan2(y_speed, x_speed))
    return direction, speed


def main():
    rocket = False  # False - kolize se Zemí, True - raketa zničí asteroid

    """
    Otevře se soubor a všechna data o asteroidech se uloží do seznamu objektů
    asteroid, taky se vytvoří seznam hmotností
    a seznam materiálů, podle těchto dvou se vytvoří a uloží histogramy
    """
    f = open('asteroidy.json')
    data = json.load(f)
    asteroids = []
    masses = []
    materials = []
    for line in data:
        if line["name"] != "Zeme":
            asteroids.append(
                Asteroid(line["x"], line["y"], line["mass"], line["name"],
                         line["direction"],
                         line["speed"], line["type"], line["acceleration"]))
            masses.append(line["mass"])
            materials.append(line["type"])
        else:
            earth = Asteroid(line["x"], line["y"], line["mass"], line["name"],
                             line["direction"], line["speed"],
                             line["type"], line["acceleration"])

    # Vytvoření histogramů
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    bins1 = round(math.sqrt(len(asteroids)))
    ax1.hist(masses, bins=bins1, edgecolor="black", rwidth=0.8,
             color="#0096FF")
    ax1.grid(axis="y", alpha=0.65)
    ax1.set_title('Mass')
    ax1.set_xlabel('Mass')
    ax1.set_ylabel('Amount of Asteroids')
    ax2.hist(materials, bins=[-0.5, 0.5, 1.5, 2.5], edgecolor="black",
             align='mid', rwidth=0.9, color="#0096FF")
    ax2.grid(axis="y", alpha=0.65)
    ax2.set_title('Materials')
    ax2.set_xlabel('Material')
    ax2.set_ylabel('Amount of Asteroids')
    plt.savefig('histogramy.png')
    plt.close()
    plt.style.use("seaborn-v0_8")
    fig, ax = plt.subplots(figsize=(9, 9))
    fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9)
    ax.set_xlim(0, SIZE_X)
    ax.set_ylim(0, SIZE_Y)
    collided: List[str] = []  # nazvy asteroidy ktere narazi do Zeme
    filtered_asteroids = []
    for asteroid in asteroids:  # filtrace asteroidů pod 500 kg
        if asteroid.mass > 500:
            filtered_asteroids.append(asteroid)
    print("Podmínky pro relevanci asteroidu splňuje", len(filtered_asteroids),
          "asteroidů.")
    """
    vytvoří se sactter, na který se podel seznamů asteroidů asteroidy umístí,
    ke každému z nich se také přidá jaho jméno
    jako text, přidá se i země, ta je uložena na knci seznamu asteroidů
    """
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    asteroids_x = []
    asteroids_y = []
    size = []
    colors = []
    names = []
    asteroids.append(earth)
    for asteroid in asteroids:
        asteroids_x.append(asteroid.x)
        asteroids_y.append(asteroid.y)
        size.append(7)
        colors.append("gray")
        names.append(asteroid.name)
        if asteroid.name != "Zeme":
            speed_x, speed_y = direction_and_speed_to_x_speed_and_y_speed(
                asteroid.direction, asteroid.speed)
            asteroid.xspeed = speed_x
            asteroid.yspeed = speed_y
    colors[-1] = "blue"
    size[-1] = 50
    scat = ax.scatter(asteroids_x, asteroids_y, c=colors, s=size,
                      edgecolors="black")
    for spine in ax.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(2)
    texty = []
    raketa = None
    if rocket:
        raketa = Rocket()
    exploded = None
    for asteroid in asteroids:
        if asteroid.name == "Zeme":
            textik = ax.text(asteroid.x, asteroid.y + 0.7, asteroid.name,
                             fontsize=6, ha='center', va='bottom')
        else:
            textik = ax.text(asteroid.x, asteroid.y, asteroid.name, fontsize=6,
                             ha='left', va='bottom')
        texty.append(textik)

    def animation_update(_, exploded, raketa):
        asteroids_y_offset = []
        asteroids_x_offset = []
        exploded_scat = None
        rocket_scat = None
        """
        pro každý asteroid se spočítá jak na něj působí každý objekt ve
        vesmíru, včetně země, případně rakety, každý výpočet s raketou první
        kontroluje, jestli existuje/nevybouchla. Efekt gravitace
        bere v potaz předešlou rychlostXY a vždy ji přepočítá na novou a na
        konec asteroid o ni posune
        """
        for k in range(len(asteroids)):
            if asteroids[k].name != "Zeme" and not asteroids[k].exploded:
                for asteroid1 in asteroids:
                    if asteroid1.name != asteroids[k].name:
                        speed_x2, speed_y2 = calculate_gravity_affected_speed(
                            asteroid1.x, asteroid1.y, asteroid1.mass,
                            asteroids[k].x, asteroids[k].y,
                            asteroids[k].xspeed, asteroids[k].yspeed)
                        asteroids[k].xspeed = speed_x2
                        asteroids[k].yspeed = speed_y2
                if raketa is not None:
                    if not raketa.exploded:
                        speed_x2, speed_y2 = calculate_gravity_affected_speed(
                            raketa.x, raketa.y, raketa.mass,
                            asteroids[k].x, asteroids[k].y,
                            asteroids[k].xspeed, asteroids[k].yspeed)
                        asteroids[k].xspeed = speed_x2
                        asteroids[k].yspeed = speed_y2
            if asteroids[k].name != "Zeme" and not asteroids[k].exploded:
                asteroids[k].x = asteroids[k].x + asteroids[k].xspeed
                asteroids[k].y = asteroids[k].y + asteroids[k].yspeed
                texty[k].set_position(
                    (texty[k].get_position()[0] + asteroids[k].xspeed,
                     texty[k].get_position()[1] + asteroids[k].yspeed))
        asteroids[-1].x = 50.0  # země zůstane fixní
        asteroids[-1].y = 50.0
        """
        Pokud se nějaký asteroid přiblíží k Zemi na 0.5 (vzdálenost přímky
        asteroid-země), označí se, že má vybouchnout
        a vypíše se do konzole a přidá do collided
        """
        for asteroid in asteroids:
            if (math.sqrt(abs(asteroid.x - earth.x) ** 2 + abs(
                    asteroid.y - earth.y) ** 2) <= 0.5
                    and asteroid.name != "Zeme"):
                if not asteroid.exploded:
                    print("Do země narazil:", asteroid.name)
                    asteroid.exploded = True
                    asteroid.explosion_pos = (asteroid.x, asteroid.y)
                    collided.append(asteroid.name)
            asteroids_x_offset.append(asteroid.x)
            asteroids_y_offset.append(asteroid.y)
            """
            jelikož bylo zadáno, že to země narazí tenhle asteroid a nebude
             použit jiný dataset, raketa se updatuje hned po pohybu tohoto
            asteroidu pokud se asteroid přibllíží k zemi (vzdálenost, kde mezi
            zemí a asteroidem není žádný jiný asteroid), tak se vygeneruje
            raketa ve snímku 0, v každém dalším snímku se raketa bude
            pohybovat, hlavně půjde o vliv gravitace, první zrychlení jen určí
            směr (zrychlí jen jednou - na počáteční rychlost, aby se dostala od
            středu země, kde kvůli vyjímce (dělení nulou) gravitaci nepočítám)
            po tomto jediném zrychlení se vypočítá palivo (3510 - 3500),
            tyto hodnoty jsou už zapsané ve třídš raketa,
            jelikož raketa má zničit explicitně tenhle asteroid
            pokud se raketa přiblíží k JAKÉMUKOLIV asteroidu  na blíže než 0.5,
            tak vybouchne a zničí ho
            """
            if asteroid.name == "A-D81HE" and raketa is not None:
                if (raketa.move_frame == 0 and
                        math.sqrt(abs(asteroid.x - earth.x) ** 2 + abs(
                            asteroid.y - earth.y) ** 2) < 13):
                    rocket_scat = ax.scatter(50, 50, s=3, zorder=6,
                                             edgecolors="black")
                    text_r = ax.text(raketa.x, raketa.y, "raketa", fontsize=6,
                                     ha='left', va='bottom')
                    texty.append(text_r)
                    rocket_scat.set_facecolors("red")
                    raketa.move_frame = 1
                    raketa.fuel += (raketa.og_mass - raketa.mass)
                if raketa.move_frame > 0 and not raketa.exploded:
                    speed_rx, speed_ry = (
                        direction_and_speed_to_x_speed_and_y_speed(
                            raketa.direction, raketa.speed))
                    raketa.xspeed = speed_rx
                    raketa.yspeed = speed_ry
                    for asteroid1 in asteroids:
                        speed_rx, speed_ry = calculate_gravity_affected_speed(
                            asteroid1.x, asteroid1.y, asteroid1.mass, raketa.x,
                            raketa.y, raketa.xspeed, raketa.yspeed)
                        raketa.xspeed = speed_rx
                        raketa.yspeed = speed_ry
                    raketa.x += raketa.xspeed
                    raketa.y += raketa.yspeed
                    raketa.direction, raketa.speed \
                        = xy_speed_to_direction_and_speed(
                            raketa.xspeed, raketa.yspeed)
                    rocket_scat = ax.scatter(raketa.x, raketa.y, s=3, zorder=6)
                    rocket_scat.set_facecolors("red")
                    texty[-1].set_position(
                        (texty[-1].get_position()[0] + raketa.xspeed,
                         texty[-1].get_position()[1] + raketa.yspeed))
                    raketa.move_frame += 1
                    for asteroid in asteroids:
                        if raketa is not None:
                            if (math.sqrt(
                                    abs(asteroid.x - raketa.x) ** 2 + abs(
                                        asteroid.y - raketa.y) ** 2)
                                    <= 0.5
                                    and asteroid.name != "Zeme"
                                    and not asteroid.exploded):
                                print("Raketa spotřebovala", raketa.fuel,
                                      "paliva a narazila do:", asteroid.name)
                                asteroid.exploded = True
                                asteroid.explosion_pos = (
                                    asteroid.x, asteroid.y)
                                raketa.exploded = True
                                texty[-1].set_text("")
            scat.set_offsets(
                np.column_stack((asteroids_x_offset, asteroids_y_offset)))

        for k in range(len(asteroids)):  # animace výbuchu
            if asteroids[k].exploded:
                asteroids[k].x = asteroids[k].explosion_pos[0]
                asteroids[k].y = asteroids[k].explosion_pos[1]
                texty[k].set_text("")
                scat._sizes[k] = 0
                if asteroids[k].explosion_frame <= 4:
                    exploded_scat = ax.scatter([asteroids[k].x],
                                               [asteroids[k].y],
                                               s=asteroids[k]
                                               .explosion_frame * 4,
                                               zorder=5, linewidths=0.4,
                                               facecolor="orange",
                                               edgecolors="red")
                    asteroids[k].explosion_frame += 1
                elif asteroids[k].explosion_frame <= 7:
                    exploded_scat = ax.scatter([asteroids[k].x],
                                               [asteroids[k].y], s=asteroids[k]
                                               .explosion_frame * 4,
                                               zorder=5,
                                               linewidths=asteroids[k]
                                               .explosion_frame * 0.2,
                                               facecolor="none",
                                               edgecolors="orange")
                    asteroids[k].explosion_frame += 1
                else:
                    exploded_scat = None
        if exploded_scat is not None:
            return scat, *texty, exploded_scat
        if rocket_scat is not None:
            return scat, *texty, rocket_scat
        return scat, *texty

    _ = FuncAnimation(fig, func=animation_update, fargs=(exploded, raketa,),
                        frames=1000, interval=0, blit=True)
    plt.show()


if __name__ == '__main__':
    main()
