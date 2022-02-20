from typing import Tuple, Dict, List, TypeVar, Optional
import json
import heapq
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import time
import logging


folder = 'a_star_logs_and_results'
timestr = time.strftime("%Y%m%d-%H%M%S")
logging.basicConfig(filename=folder + '/global_route_' +
                    timestr + '.log', level=logging.DEBUG)
# logging.disable()
logging.debug("Los gehts!")

start = time.time()

tracks_used = {"default": 1, "1w2s": 5, "1w2s_shield": 3}
capacity = {'M6': 65, 'M7': 65, 'M8': 39, 'M9': 39, 'M10': 16, 'M11': 16}
usable_cap = 0.8
preference = {"FAST_CLOCK": 1, "HIGH_SPEED_DATA": 2, "CLOCK": 3, "DATA": 4}
num_used_tracks = {}
upper_layer = {"FAST_CLOCK": 11, "HIGH_SPEED_DATA": 9, "CLOCK": 9, "DATA": 8}
lower_layer = {"FAST_CLOCK": 10, "HIGH_SPEED_DATA": 8, "CLOCK": 7, "DATA": 6}


class FileOperations:
    def __init__(self, fileName=None):
        self.fileName = fileName

    def write_to_jsonfile(self, fileName, writedata):
        with open(fileName, 'w') as outfile:
            json.dump(writedata, outfile)
        return outfile


GridLocation = Tuple[int, int, int]
CellLocation = Tuple[int, int]
Location = TypeVar("Location")


class ExpandWavefront:
    def __init__(self, num_layers, num_cols, num_rows, gcell_weight) -> None:
        self.num_layers = num_layers
        self.num_cols = num_cols
        self.num_rows = num_rows
        self.gcell_weight = gcell_weight

    def in_bounds(self, id: GridLocation) -> bool:
        (l, x, y) = id
        return (
            min_layer <= l <= max_layer
            and 0 <= x < self.num_cols
            and 0 <= y < self.num_rows
        )

    def neighbors_multilayer(self, id: GridLocation):
        (l, x, y) = id

        if l % 2 == 0:
            if l == max_layer:
                neighbors = [(l, x + 1, y), (l, x - 1, y),
                             (l - 1, x, y)]
            elif l == min_layer:
                neighbors = [(l, x + 1, y), (l, x - 1, y),
                             (l + 1, x, y)]
            else:
                neighbors = [(l, x + 1, y), (l, x - 1, y),
                             (l + 1, x, y), (l - 1, x, y)]
        elif l % 2 == 1:
            if l == max_layer:
                neighbors = [(l, x, y + 1), (l, x, y - 1),
                             (l - 1, x, y)]
            elif l == min_layer:
                neighbors = [(l, x, y + 1), (l, x, y - 1),
                             (l + 1, x, y)]
            else:
                neighbors = [(l, x, y + 1), (l, x, y - 1),
                             (l + 1, x, y), (l - 1, x, y)]

        results = filter(self.in_bounds, neighbors)
        return results


class PriorityQueue:
    def __init__(self):
        self.elements: List = []

    def empty(self) -> bool:
        return not self.elements

    def put(self, item, priority: float):
        heapq.heappush(self.elements, (priority, item))

    def get(self):
        return heapq.heappop(self.elements)[1]


class WeightedEdge(ExpandWavefront):
    def __init__(self, num_layers: int, num_cols: int, num_rows: int, gcell_weight: int):
        super().__init__(num_layers, num_cols, num_rows, gcell_weight)
        self.weights: Dict[GridLocation, float] = {}

    def cost(self, to_node: GridLocation) -> float:
        return self.weights.get(to_node, self.gcell_weight)


def heuristic(a: GridLocation, b: GridLocation) -> float:
    (l1, x1, y1) = a
    (l2, x2, y2) = b
    return abs(x1 - x2) + abs(y1 - y2) + abs(l1 - l2)


def a_star_search(graph: WeightedEdge, start: Location, goal: Location):
    wavefront = PriorityQueue()
    wavefront.put(start, 0)
    came_from: Dict[Location, Optional[Location]] = {}
    cost_so_far: Dict[Location, float] = {}
    came_from[start] = None
    cost_so_far[start] = 0
    timeout = 5
    timeout_start = time.time()

    flag = True
    while not wavefront.empty():
        if (time.time() > timeout_start + timeout) and flag == True:
            logging.debug("TIMEDOUT::::::::" + str(start) + str(goal))
            flag = False

        current: Location = wavefront.get()

        if current == goal:
            break

        for next in graph.neighbors_multilayer(current):
            if next in num_used_tracks.keys():
                if num_used_tracks[next] > usable_cap * capacity['M' + str(next[0])]:
                    new_cost = cost_so_far[current] + \
                        graph.cost(next) + 5 * num_used_tracks[next]
                else:
                    new_cost = cost_so_far[current] + graph.cost(next)
            else:
                new_cost = cost_so_far[current] + graph.cost(next)

            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                priority = new_cost + heuristic(next, goal)
                wavefront.put(next, priority)
                came_from[next] = current
    logging.debug("TIME TAKEN:::::" + str(time.time() - timeout_start))
    return came_from, cost_so_far


def reconstruct_path(came_from: Dict[Location, Location], start: Location, goal: Location) -> List[Location]:
    current: Location = goal
    path: List[Location] = []

    while current != start:
        path.append(current)
        current = came_from[current]
    path.append(start)
    path.reverse()
    return path


def sorted(wires_list):
    prioritized_wireslist = PriorityQueue()
    pwl = []
    for w in wires_list:
        if w[2] in preference.keys():
            prioritized_wireslist.put(w, preference[w[2]])
        else:
            prioritized_wireslist.put(w, 3)
    for x in prioritized_wireslist.elements:
        pwl.append(x[1])
    return pwl


def create_Fig(initial_image):
    size = np.shape(initial_image)

    # create the figure
    fig = plt.figure()

    # create the axis object:
    ax = fig.add_subplot(111)

    # draw the initial image as a heat map without interpolation
    im = ax.imshow(initial_image, interpolation="none", cmap='gnuplot2')

    # Fixing the grid:
    # major ticks every 20, minor ticks every 5
    x_ticks = np.arange(-0.5, size[1]+.5, 1)
    y_ticks = np.arange(-0.5, size[0]+.5, 1)

    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)

    # and a corresponding grid
    ax.grid(which='both')

    # Turn off tick labels
    ax.set_yticklabels([])
    ax.set_xticklabels([])

    # Show the initial image
    plt.show(block=False)

    print("please press enter to proceed ...")
    input()
    return fig, im


def visualize(paths):
    coordinates = []

    k = 6
    while k <= 15:
        for j in range(num_rows):
            for i in range(num_cols):
                coordinates.append((k, i, j))
        k += 1

    grid = pd.DataFrame(
        [],
        index=pd.MultiIndex.from_tuples(coordinates, names=["L", "X", "Y"]),
        columns=["reached"]
    )

    grid["reached"] = 0
    for i in range(len(paths)):
        for j in paths[i]:
            grid.at[j, "reached"] = i+1

    '''
    image1 = []
    k = 10
    while k < 12:
        for j in range(num_rows):
            for i in range(num_cols):
                # print(i)
                image1.append(grid.at[(k, i, j), "reached"])
        k += 1

    B = np.reshape(image1, (-1, num_cols))
    # print(B)
    np.savetxt(r"grid_data.txt", B, fmt="%d")
    '''
    image2 = []
    for j in range(num_rows-1, 0, -1):
        for i in range(num_cols):
            image2.append(grid.at[(6, i, j), "reached"])
    f2, i2 = create_Fig(np.reshape(image2, (-1, num_cols)))
    X2 = np.reshape(image2, (-1, num_cols))
    #fig, axs = plt.subplots(ncols=1, nrows=2)
    #sns.heatmap(X2, ax=axs[0][0]).invert_yaxis()

    image31 = []
    for j in range(num_rows-1, 0, -1):
        for i in range(num_cols):
            image31.append(grid.at[(7, i, j), "reached"])
    f3, i3 = create_Fig(np.reshape(image31, (-1, num_cols)))
    #X31 = np.reshape(image31, (-1, num_cols))
    #sns.heatmap(X31, ax=axs[0][1]).invert_yaxis()

    image3 = []
    for j in range(num_rows-1, 0, -1):
        for i in range(num_cols):
            image3.append(grid.at[(8, i, j), "reached"])
    f4, i4 = create_Fig(np.reshape(image3, (-1, num_cols)))
    # X3 = np.reshape(image3, (-1, num_cols))
    # sns.heatmap(X3, ax=axs[0][2]).invert_yaxis()
    # a2.invert_yaxis()

    image4 = []
    for j in range(num_rows-1, 0, -1):
        for i in range(num_cols):
            image4.append(grid.at[(9, i, j), "reached"])
    f5, i5 = create_Fig(np.reshape(image4, (-1, num_cols)))
    # X4 = np.reshape(image4, (-1, num_cols))
    # sns.heatmap(X4, ax=axs[0][3]).invert_yaxis()

    image5 = []
    for j in range(num_rows-1, 0, -1):
        for i in range(num_cols):
            image5.append(grid.at[(10, i, j), "reached"])
    f6, i6 = create_Fig(np.reshape(image5, (-1, num_cols)))
    # X5 = np.reshape(image5, (-1, num_cols))
    # sns.heatmap(X5, ax=axs[0][4]).invert_yaxis()
    print("LAST ONE")
    image6 = []
    for j in range(num_rows-1, 0, -1):
        for i in range(num_cols):
            image6.append(grid.at[(11, i, j), "reached"])
    f7, i7 = create_Fig(np.reshape(image6, (-1, num_cols)))
    # X6 = np.reshape(image6, (-1, num_cols))
    # sns.heatmap(X6, ax=axs[1][0]).invert_yaxis()

    plt.xlim(0, num_cols)
    plt.ylim(0, num_rows)
    # plt.show()


# Actual implementation starts here
input_json_file = "final_output.json"
global weight


# macros_list = []
with open(input_json_file) as f:
    data = json.load(f)
    wires_list = []
    chip_dimensions = data["chip_dimensions"]
    num_layers = data["num_of_layers"]
    weight = data["edge_weight"]
    num_cols = int(
        (float(chip_dimensions['UR_X']) - float(chip_dimensions['LL_X'])) / 5) + 1
    num_rows = int(
        (float(chip_dimensions['UR_Y']) - float(chip_dimensions['LL_Y'])) / 5) + 1

    for inp in data['netlist']:
        in_pin_name = list(inp.keys())[0]
        in_x = int(inp[in_pin_name]['x_'+in_pin_name]/5)

        in_y = int(inp[in_pin_name]['y_'+in_pin_name]/5)
        print("INPIN:::::", in_x, in_y)
        in_layer = int(inp[in_pin_name]['layer'])

        out_pin_name = list(inp.keys())[1]
        out_x = int(inp[out_pin_name]['x_'+out_pin_name]/5)
        out_y = int(inp[out_pin_name]['y_'+out_pin_name]/5)
        out_layer = int(inp[out_pin_name]['layer'])
        print("OUTPIN:::::", out_x, out_y)

        net_type = inp['net_type']
        ndr_type = inp['ndr']
        print("net_type:::::", net_type)

        in_tuple = (in_layer, in_x, in_y)
        out_tuple = (out_layer, out_x, out_y)

        wires_list.append([in_tuple, out_tuple, net_type,
                          (out_pin_name, in_pin_name), ndr_type])

# pins_list = [[(11, 0, 89), (10, 523, 336)], [(11, 248, 86), (10, 149, 0)], [(11, 248, 86), (10, 149, 0)], [(11, 248, 86), (10, 149, 0)], [(11, 248, 86), (10, 149, 0)]]
# pins_list = [[(11, 327, 345), (10, 149, 0)]]
#logging.debug("WIRES LIST FOR" + str(len(wires_list)) + ":::::::::::::::::::" + str(wires_list))
pins_list = sorted(wires_list)
print(pins_list[:10])
pins_list = wires_list[:100]
# print(pins_list)


paths = []
graph_with_weights = WeightedEdge(
    num_layers=num_layers, num_cols=num_cols, num_rows=num_rows, gcell_weight=weight)

for i, pin in enumerate(pins_list):
    # print("******************************PIN*************************", pin)
    maxi_layer = max(pin[0][0], pin[-4][0])
    print("WIRE NUMBER:::: " + str(i+1) + " OUT OF " + str(len(pins_list)))
    print("PIN=>", pin)
    pins_list[i] = pin
    logging.debug("WIRE NUMBER:::: " + str(i+1) +
                  " OUT OF " + str(len(pins_list)))
    logging.debug("PIN::::::" + str(pin))
    ndr_type = pin[-1]
    net_type = pin[-3]
    max_layer = upper_layer[net_type]
    min_layer = lower_layer[net_type]
    path = []
    for n in range(len(pin)-4):
        # print("PIN:", pin)
        logging.debug("A-STAR START::::: " + str(pin[n]) + str(pin[n+1]) + str(
            ' MAX LAYER::') + str(max_layer) + str(' MIN LAYER::') + str(min_layer))

        local_source = pin[n]
        local_target = pin[n+1]

        came_from, cost_so_far = a_star_search(
            graph_with_weights, local_source, local_target)
        # print("A STAR END")
        #print("RECONSTRUCT PATH START  " + str(n))
        # print("============>", came_from, pin[n], pin[n+1])
        pat = reconstruct_path(
            came_from, start=local_source, goal=local_target)
        # print("RECONSTRUCT PATH END")
        path.extend(pat)

        for p in path[1:-1]:
            if ndr_type not in tracks_used.keys():
                ndr_type = "default"
            if p not in num_used_tracks.keys():
                num_used_tracks[p] = 0
            num_used_tracks[p] += tracks_used[ndr_type]
    # print(num_used_tracks)

    paths.append(path)

# print(paths)
end = time.time()
print("Time taken is:", (end-start)/60)

#time_stamp = datetime.date.today()

'''
GR_bucket_data = {}
GR_bucket_data["GR_bucket"] = []

for j in range(len(paths)):
    id, ft_input_name, net_type, net_coordinates = j, pins_list[
        j][-2], pins_list[j][-3], paths[j]
    input_pin_coordinate = pins_list[j][0]
    output_pin_coordinate = pins_list[j][-4]
    current_gr_bucket = {}
    current_gr_bucket["id"] = j+1
    current_gr_bucket["ft_input_pin"] = ft_input_name[1]
    current_gr_bucket["input_pin_coordinate"] = input_pin_coordinate
    current_gr_bucket["ft_outpin"] = ft_input_name[0]
    current_gr_bucket["output_pin_coordinate"] = output_pin_coordinate
    current_gr_bucket["net_type"] = net_type
    current_gr_bucket["net_coordinates"] = net_coordinates
    GR_bucket_data["GR_bucket"].append(current_gr_bucket)
    fileOperations.write_to_jsonfile(folder + "/GR_bucket_" + folder + "_" + timestr + ".json", GR_bucket_data)
'''

textfile = open(folder + "/GR_bucket_" +
                folder + "_" + timestr + ".txt", "w")
for j in range(len(paths)):
    textfile.write(str(j+1)+"\t")
    textfile.write(pins_list[j][-1]+"\t")
    textfile.write(pins_list[j][-2][0] + '\t' + pins_list[j][-2][1] + '\n')

    for element in paths[j]:
        # for i in paths[j]:
        for i in element:
            textfile.write(str(i) + "\t")
        textfile.write("\n")
        # wn+=1
    textfile.write(str(0)+"\n")
textfile.close()

visualize(paths)
