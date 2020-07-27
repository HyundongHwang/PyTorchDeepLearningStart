g_oldLogType = "none"  # none, single, multi


def log(name, value):
    global g_oldLogType
    valueStr = "{}".format(value)

    if "\n" in valueStr:
        if g_oldLogType != "multi":
            print("")
        g_oldLogType = "multi"
        print("{} : ".format(name))
        print("    {}".format(valueStr))
        print("")
    else:
        g_oldLogType = "single"
        print("{} : {}".format(name, valueStr))


def model_to_str(model):
    resStr = ""

    for param in model.parameters():
        resStr += "    {} {}\n".format(param.shape, param.data)

    return resStr


def log_model(name, model):
    print("")
    print("{} : ".format(name))

    for param in model.parameters():
        print("    {} {}".format(param.shape, param.data))

    print("")
