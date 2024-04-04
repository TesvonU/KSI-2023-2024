import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.collections import LineCollection
from typing import List, Tuple, Set
COLOR = Tuple[float, float, float]


class ButtonFunc:
    def __init__(self):
        self.speed = [0, 10, 20, 30, 40]
        self.currrent_speed = 2
        self.output_speed = self.speed[self.currrent_speed]
        self.do_reset = False
        self.reconstruct = False

    def faster(self, event):
        if self.currrent_speed < 4:
            self.currrent_speed += 1
            self.output_speed = self.speed[self.currrent_speed]

    def slower(self, event):
        if self.currrent_speed > 0:
            self.currrent_speed += -1
            self.output_speed = self.speed[self.currrent_speed]

    def reset(self, event):
        self.do_reset = True

    def cycle(self, event):
        self.reconstruct = True


def plot_music(file: str,
               colors: List[COLOR] = None,
               linewidths: List[float] = None,
               groups: List[Set[str]] = None,
               main_instrument: int = None,
               masked: Set[int] = None,
               special: str = None,
               special_index: int = None):
    '''na začátku program přečte celý txt soubor a vytvoří pro všechna data
    slovník, kde klíč je ID, dále se vytvoří segmenty pro linecollection,
    pomocí těchto dat'''
    my_file = open(file)
    if main_instrument is not None:
        main = main_instrument
        if masked is not None:
            if main in masked:
                main = None
    else:
        main = None
    default_pie_line = "black"
    if colors is None:
        colors = [(0.0, 0.0, 0.0)]
        default_pie_line = "white"
    instruments_amount = my_file.readline()
    instruments_id_name = {}
    instruments_id_start_end = {}
    instruments_id_tone = {}
    instruments_id_duration = {}
    for k in range(int(instruments_amount)):
        temp = my_file.readline().split(";")
        instruments_id_name[int(temp[0])] = temp[1]
    for k in range(int(instruments_amount)):
        temp = my_file.readline().split(";")
        instruments_id_start_end[int(temp[0])] = (int(temp[1]), int(temp[2]))
    current_duration = [0 for k in range(int(instruments_amount))]
    for k in range(int(instruments_amount)):
        instruments_id_tone[k] = []
        instruments_id_duration[k] = []
    loop = True
    while loop:
        temp = my_file.readline()
        if temp == "" or temp == '\n':
            loop = False
        else:
            temp = temp.split(";")
            instruments_id_tone[int(temp[0])].append(int(temp[1]))
            instruments_id_tone[int(temp[0])].append(int(temp[1]))
            instruments_id_duration[int(temp[0])].append(
                current_duration[int(temp[0])])
            instruments_id_duration[int(temp[0])].append(
                int(temp[2]) + current_duration[int(temp[0])])
            current_duration[int(temp[0])] += int(temp[2])
    plots = {}
    for k in range(int(instruments_amount)):
        x = [i for i in instruments_id_duration[k]]
        y = [i for i in instruments_id_tone[k]]
        segments = [[] for _ in range(len(x) - 1)]
        for i in range(len(x) - 1):
            segments[i].append([x[i], y[i]])
            segments[i].append([x[i + 1], y[i + 1]])
        # Program zkontroluje parametry pro vzhled linií a uplatní je
        if colors is not None:
            amount_colors = len(colors) - 1
            if amount_colors < k:
                choice = k
                while choice > amount_colors:
                    choice += -(amount_colors+1)
                color_choice = colors[choice]
            else:
                color_choice = colors[k]
        else:
            color_choice = "black"
        if linewidths is not None:
            amount_widths = len(linewidths) - 1
            if amount_widths < k:
                choice = k
                while choice > amount_widths:
                    choice += -(amount_widths+1)
                widths_choice = linewidths[choice]
            else:
                widths_choice = linewidths[k]
        else:
            widths_choice = 1.0
        my_lc = LineCollection(segments, color=color_choice,
                               linewidths=widths_choice)
        my_lc.set_label(instruments_id_name[k])
        if k == main:
            status = "main"
        else:
            status = "non-main"
        '''LineCollections a další relevantní údaje k grafu se uloží do
         slovníku, dále k nim přidáme do slovníku další důležité ádaje a pak
          spojíme hodnoty pro klíče, které mají být v jendé skupině'''
        plots[k] = [my_lc, status, instruments_id_start_end[k]]
    plots = dict(sorted(plots.items()))
    end = []
    current_group = [idx for idx in instruments_id_name]
    combinations = []
    for key in plots:
        graph = []
        for coll in plots[key][:-2]:
            found = False
            for k in range(len(coll.get_segments())):
                last_x = coll.get_segments()[-(k + 1)][-1, 0]
                last_y = coll.get_segments()[-(k + 1)][-1, 1]
                if last_y != -1:
                    graph.append(last_x)
                    found = True
                    break
            if not found:
                graph.append(0)
        end.append(graph)
    for key in plots:
        plots[key].append(end[key])
    new_duration = {}
    new_tone = {}
    for key in instruments_id_duration:
        new_duration[key] = [instruments_id_duration[key][0]]\
                            + instruments_id_duration[key][1:-1:2]
    for key in instruments_id_tone:
        new_tone[key] = instruments_id_tone[key][1::2]
    ration = [0 for kx in new_duration]
    if masked is not None:
        for mask in masked:
            del plots[mask]
    if groups is not None:
        for group in groups:
            if len(group) > 1:
                to_combine = []
                new_status = "non-main"
                tone_group = []
                max_x = []
                for name in group:
                    for key in instruments_id_name:
                        if key in plots:
                            if instruments_id_name[key] == name:
                                to_combine.append(key)
                                if plots[key][1] == "main":
                                    new_status = "main"
                                tone_group += plots[key][2]
                                max_x += plots[key][-1]
                segment_combined = []
                combinations.append(to_combine)
                for id in to_combine:
                    segment_combined.append(plots[id][0])
                    del plots[id]
                segment_combined.append(new_status)
                if len(tone_group) > 0:
                    segment_combined.append([min(tone_group), max(tone_group)])
                    segment_combined.append(max(max_x))
                    plots[to_combine[0]] = segment_combined
    '''Vytvoříme okno s grafy a vytvoříme všechny subploty, které budou
    potřeba. Ty, které mají být animace Linecollectionu dáme do seznamu 
    animations_plot, ať je můžeme potom používat v cyklech. Poté do subplotů
    vložíme kolekci/kolekce ze slovníků, nastavíme legendy a x/y limit
    
    '''
    fig, ax = plt.subplots(figsize=(10, 9))
    ax.set_axis_off()
    wanted_plots = len(plots) - (main is not None)
    animation_plots = []
    if wanted_plots >= 1:
        ax1 = plt.subplot(5, 3, 1)
        animation_plots.append(ax1)
    if wanted_plots >= 2:
        ax2 = plt.subplot(5, 3, 2)
        animation_plots.append(ax2)
    if wanted_plots >= 3:
        ax3 = plt.subplot(5, 3, 3)
        animation_plots.append(ax3)
    if wanted_plots >= 4:
        ax4 = plt.subplot(5, 3, 4)
        animation_plots.append(ax4)
    if wanted_plots >= 5:
        ax5 = plt.subplot(5, 3, 5)
        animation_plots.append(ax5)
    if wanted_plots >= 6:
        ax6 = plt.subplot(5, 3, 6)
        animation_plots.append(ax6)
    if main is not None:
        ax7 = plt.subplot(5, 1, 3)
        animation_plots.append(ax7)
    if special:
        ax12 = plt.subplot(5, 3, 12)
    if special == "pie":
        ax12.pie([1, 1], colors= [(1.0, 1.0, 1.0)])
    elif special == "line":
        if colors is not None:
            amount_colors = len(colors) - 1
            if amount_colors < special_index:
                choice = special_index
                while choice > amount_colors:
                    choice += -(amount_colors + 1)
                color_choice2 = colors[choice]
            else:
                color_choice2 = colors[special_index]
        else:
            color_choice2 = "black"
        line, = ax12.plot([], [], color=color_choice2)
        ax12.set_title("Note counts per instrument")
        ax12.set_xlim(0, 500)
        ax12.set_ylim(0, 1)

    ax10 = plt.subplot(5, 3, 10)
    ax13 = plt.subplot(5, 3, 13)
    ax14 = plt.subplot(5, 3, 14)
    ax15 = plt.subplot(5, 3, 15)
    plots = dict(sorted(plots.items()))
    counter = 0
    order = [0 for x in animation_plots]
    for key in plots:
        if plots[key][-3] == "main":
            order[-1] = plots[key][-1]
            for coll in plots[key][:-3]:
                animation_plots[-1].add_collection(coll)
                animation_plots[-1].set_ylim(plots[key][-2][0],
                                             plots[key][-2][1])
                animation_plots[-1].set_xlim(0, 500)
                animation_plots[-1].legend()
        else:
            order[counter] = plots[key][-1]
            for coll in plots[key][:-3]:
                animation_plots[counter].add_collection(coll)
            animation_plots[counter].set_ylim(plots[key][-2][0],
                                              plots[key][-2][1])
            animation_plots[counter].set_xlim(0, 500)
            animation_plots[counter].legend()
            counter += 1
    plt.tight_layout()
    my_file.close()
    plt.show(block=False)
    current_note = 0
    ration_value = False
    y_percent = []
    x_time = []
    # nastavení tlačítek, jejich funkcej sou nahoře
    button_funcs = ButtonFunc()
    button_faster = Button(ax13, 'Faster')
    button_faster.on_clicked(button_funcs.faster)
    button_slower = Button(ax14, 'Slower')
    button_slower.on_clicked(button_funcs.slower)
    button_reset = Button(ax15, 'Reset')
    button_reset.on_clicked(button_funcs.reset)
    button_cycle = Button(ax10, 'Cycle')
    button_cycle.on_clicked(button_funcs.cycle)
    # vytvořeí se seznam grafů a jejich nástrojů v nich, tím se řídí cycle
    new_group = []
    flat_combo = []
    for sublist in combinations:
        for number in sublist:
            flat_combo.append(number)
    for number in current_group:
        if number in flat_combo:
            pass
        else:
            new_group.append([number])
    for sublist in combinations:
        new_group.append(sublist)
    possible_main = []
    if masked is None:
        masked =[]
    for element in current_group:
        if element not in masked:
            possible_main.append(element)
    final_group = []
    for sublist in new_group:
        sublist_1 = []
        for element in sublist:
            if element not in masked:
                sublist_1.append(element)
        if (len(sublist_1)) > 0:
            final_group.append(sublist_1)
    main2 = main_instrument
    '''notu beru jako zahranou v moment, kdy je na pravé straně grafu
     (levý limit)
    
    Ve while cyklu první zjistíme, jestli se má procyklovat main
        - vytvoří se nové LineCollectiony a dají na nové místo
    Nebo jestli se má resetovat x osa
        - x lim se změní a resetuje se special graf
    
    Pak se k x lim Collection grafů příčte hodnota nasstavená tlačítky faster/
    slower a ověří se, jestli ještě pokračují a není třeba je zastavit.
    Hodnoty, které jsou zahrané se pomocí slovníku s ID a tónem/duration a 
    pomocí counteru current_note uloží do seznamu rations, podle něj se 
    počítají special grafy.
    '''
    while True:
        if button_funcs.reconstruct:
            if main_instrument is not None:
                #plots regen
                for key in plots:
                    ky = 0
                    if isinstance(plots[key][0:-3], list):
                        for lc in plots[key][0:-3]:
                            my_lc = LineCollection(lc.get_segments(),
                                                   color=lc.get_color(),
                                                   linewidths=
                                                   lc.get_linewidth())
                            my_lc.set_label(lc.get_label())
                            plots[key][ky] = my_lc
                            ky +=1
                    else:
                        lc = plots[key][0:-3]
                        my_lc = LineCollection(lc.get_segments(),
                                               color=lc.get_color(),
                                               linewidths=lc.get_linewidth())
                        my_lc.set_label(lc.get_label())
                        plots[key][ky] = my_lc
                        ky +=1
                #plots regen
                if main2 == (len(instruments_id_name)-1):
                    main2 = 0
                else:
                    main2 += 1
                while main2 not in possible_main:
                    if main2 == (len(instruments_id_name) - 1):
                        main2 = 0
                    else:
                        main2 += 1
                for sublist in final_group:
                    for element in sublist:
                        if element == main2:
                            main_group = sublist
                for key in plots:
                    plots[key][-3] = "non-main"
                    if key in main_group:
                        plots[key][-3] = "main"
                x_lim_storage = []
                for ani_plot in animation_plots:
                    x_lim_storage.append(ani_plot.get_xlim())
                    ani_plot.clear()
                animation_plots = []
                if wanted_plots >= 1:
                    ax1 = plt.subplot(5, 3, 1)
                    animation_plots.append(ax1)
                if wanted_plots >= 2:
                    ax2 = plt.subplot(5, 3, 2)
                    animation_plots.append(ax2)
                if wanted_plots >= 3:
                    ax3 = plt.subplot(5, 3, 3)
                    animation_plots.append(ax3)
                if wanted_plots >= 4:
                    ax4 = plt.subplot(5, 3, 4)
                    animation_plots.append(ax4)
                if wanted_plots >= 5:
                    ax5 = plt.subplot(5, 3, 5)
                    animation_plots.append(ax5)
                if wanted_plots >= 6:
                    ax6 = plt.subplot(5, 3, 6)
                    animation_plots.append(ax6)
                if main is not None:
                    ax7 = plt.subplot(5, 1, 3)
                    animation_plots.append(ax7)
                plots = dict(sorted(plots.items()))
                counter = 0
                order = [0 for x in animation_plots]
                for key in plots:
                    if plots[key][-3] == "main":
                        order[-1] = plots[key][-1]
                        for coll in plots[key][:-3]:
                            animation_plots[-1].add_collection(coll)
                            animation_plots[-1].set_ylim(plots[key][-2][0],
                                                         plots[key][-2][1])
                            animation_plots[-1].set_xlim(x_lim_storage[-1])
                            animation_plots[-1].legend()
                    else:
                        order[counter] = plots[key][-1]
                        for coll in plots[key][:-3]:
                            animation_plots[counter].add_collection(coll)
                        animation_plots[counter].set_ylim(plots[key][-2][0],
                                                          plots[key][-2][1])
                        animation_plots[counter].set_xlim(
                            x_lim_storage[counter])
                        animation_plots[counter].legend()
                        counter += 1
            button_funcs.reconstruct = False
            fig.canvas.draw()
            fig.canvas.flush_events()
        else:
            if button_funcs.do_reset:
                print(animation_plots)
                for k in range(len(animation_plots)):
                    animation_plots[k].set_xlim(0, 500)
                y_percent = []
                x_time = []
                current_note = 0
                ration_value = False
                ration = [0 for kx in new_duration]
                new_duration = {}
                new_tone = {}
                for key in instruments_id_duration:
                    new_duration[key] = [instruments_id_duration[key][0]] + \
                                        instruments_id_duration[key][1:-1:2]
                for key in instruments_id_tone:
                    new_tone[key] = instruments_id_tone[key][1::2]
                if special == "line":
                    ax12.set_xlim(0, 500)
                if special == "pie" or special == "bar":
                    ax12.clear()
                if special == "pie":
                    ax12.pie([1, 1], colors=[(1.0, 1.0, 1.0)])
                fig.canvas.draw()
                fig.canvas.flush_events()
                button_funcs.do_reset = False
            else:
                cahanged = False
                for k in range(len(animation_plots)):
                    if order[k] > animation_plots[k].get_xlim()[0]:
                        animation_plots[k].set_xlim(
                            animation_plots[k].get_xlim()[0]
                            + button_funcs.output_speed,
                            animation_plots[k].get_xlim()[1]
                            + button_funcs.output_speed)
                        cahanged = True
                if cahanged and current_note > 500 and special == "line":
                    ax12.set_xlim(ax12.get_xlim()[0]
                                  + button_funcs.output_speed,
                                  ax12.get_xlim()[1]
                                  + button_funcs.output_speed)
                for key in new_duration:
                    if len(new_duration[key]) > 0:
                        if new_duration[key][0] <= current_note:
                            new_duration[key].pop(0)
                            played_tone = new_tone[key].pop(0)
                            if played_tone != -1 and played_tone != "-1":
                                ration[key] += 1
                                ration_value = True
                if special == "pie":
                    if ration_value:
                        ax12.clear()
                        ax12.pie(ration, colors=colors, autopct='%1.1f%%',
                                 wedgeprops=dict(linewidth=1,
                                                 edgecolor=default_pie_line),
                                 textprops=dict(color='white'))
                        ax12.set_title("Note counts per instrument")
                if special == "bar":
                    if ration_value:
                        ax12.clear()
                        ax12.bar([("ID " + str(key)) for key in new_duration],
                                 ration, color=colors)
                        ax12.set_title("Note counts per instrument")
                    if not ration_value:
                        ax12.set_ylim(0, 4)
                if special == "line":
                    x_time.append(current_note)
                    y_percent.append(ration[special_index] /
                                     max(sum(ration), 1))
                    line.set_data(x_time, y_percent)
                current_note += button_funcs.output_speed
                fig.canvas.draw()
                fig.canvas.flush_events()



