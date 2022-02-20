import json
from random import uniform, randint

# JSON file
f = open('sample.json', "r")

# Reading from file
data = json.loads(f.read())


def askTheUser():
    num_layers = int(input("Enter the number of layer: "))
    via_penalty = int(input("Enter the penalty: "))
    edge_weight = int(input("Enter the edge weight: "))
    print("Now enter the chip dimensions")
    ll_user_x = float(
        input("Enter lower bound for X includes float numbers as well: "))
    ur_user_x = float(
        input("Enter upper bound for X includes float numbers as well: "))
    ll_user_y = float(
        input("Enter lower bound for Y includes float numbers as well: "))
    ur_user_y = float(
        input("Enter upper bound for Y includes float numbers as well: "))
    return num_layers, via_penalty, edge_weight, ll_user_x, ur_user_x, ll_user_y, ur_user_y


i = 4

possible_netclass = ["DATA", "CLOCK", "FAST_CLOCK", "HIGH_SPEED_DATA"]
possible_ndr = {"DATA": 'default', "CLOCK": 'default',
                "HIGH_SPEED_DATA": '1w2s', "FAST_CLOCK": '1w2s_shield'}

preferred_layers = {"DATA": [6, 7, 8], "CLOCK": [7, 8, 9],
                    "HIGH_SPEED_DATA": [8, 9, 9], "FAST_CLOCK": [10, 11, 11]}


def generateNetList(idx, ll_x, ur_x, ll_y, ur_y):
    generatePin = {}
    generatePin["pin_in_" + str(idx)] = {}
    generatePin["pin_out_" + str(idx)] = {}
    x_pin_in_4 = uniform(ll_x, ur_x)
    y_pin_in_4 = uniform(ll_y, ur_y)

    x_pin_out_4 = uniform(ll_x, ur_x)
    y_pin_out_4 = uniform(ll_y, ur_y)

    generatePin["pin_in_" + str(idx)]["x_pin_in_" +
                                      str(idx)] = round(x_pin_in_4, 3)
    generatePin["pin_in_" + str(idx)]["y_pin_in_" +
                                      str(idx)] = round(y_pin_in_4, 3)

    generatePin["pin_out_" + str(idx)]["x_pin_out_" +
                                       str(idx)] = round(x_pin_out_4, 3)
    generatePin["pin_out_" + str(idx)]["y_pin_out_" +
                                       str(idx)] = round(y_pin_out_4, 3)

    generatePin["net_property"] = possible_netclass[randint(0, 3)]
    generatePin["pin_in_" +
                str(idx)]["layer"] = preferred_layers[generatePin["net_property"]][randint(0, 2)]
    generatePin["pin_out_" +
                str(idx)]["layer"] = preferred_layers[generatePin["net_property"]][randint(0, 2)]
    generatePin["ndr"] = possible_ndr[generatePin["net_property"]]
    data["netlist"].append(generatePin)


num_layers, via_penalty, edge_weight, ll_user_x, ur_user_x, ll_user_y, ur_user_y = askTheUser()
data["num_of_layers"] = num_layers
data["via_penalty"] = via_penalty
data["edge_weight"] = edge_weight
data["chip_dimensions"]["LL_X"] = ll_user_x
data["chip_dimensions"]["UR_X"] = ur_user_x
data["chip_dimensions"]["LL_Y"] = ll_user_y
data["chip_dimensions"]["UR_Y"] = ur_user_y

while(i < 1001):
    generateNetList(i, ll_user_x, ur_user_x, ll_user_y, ur_user_y)
    i = i+1

with open('final_output.json', 'w') as f:
    json.dump(data, f)
