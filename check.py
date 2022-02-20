import json

with open("final_output.json") as f:
    data = json.load(f)
    # wires_list = []
    speedclasses = data["net_properties"]
    print(speedclasses)
    chip_dimensions = data["chip_dimensions"]
    print("chip_dimensions:::", chip_dimensions)
    weight = data["edge_weight"]
    print("weight::", weight)
    num_cols = int((float(chip_dimensions['UR_X']) - float(
        chip_dimensions['LL_X'])) / 5) + 1
    print("NUM OF COLUMNS::::::::", num_cols)
    num_rows = int((float(chip_dimensions['UR_Y']) - float(
        chip_dimensions['LL_Y'])) / 5) + 1
    print("NUM OF ROWS::::::::", num_rows)

    for inp in data['netlist']:
        in_pin_name = list(inp.keys())[0]
        #print("****************PIN NAME******************", in_pin_name)
        # print(inp)
        in_x = int(inp[in_pin_name]['x_'+in_pin_name]/5)
        #print("in_x:::::", in_x)
        in_y = int(inp[in_pin_name]['y_'+in_pin_name]/5)
        print("INPIN:::::", in_x, in_y)

        out_pin_name = list(inp.keys())[1]
        #print("****************PIN NAME******************", out_pin_name)
        # print(inp)
        out_x = int(inp[out_pin_name]['x_'+out_pin_name]/5)
        out_y = int(inp[out_pin_name]['y_'+out_pin_name]/5)
        print("OUTPIN:::::", out_x, out_y)

        net_property = inp['net_property']
        print("net_property:::::", net_property)
        
        
