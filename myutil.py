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


def tensor_to_str(tensor):
    resStr = "{} {}".format(tensor.shape, tensor.data)
    return resStr

def model_to_str(model):
    resStr = ""

    for param in model.parameters():
        resStr += "    {}\n".format(tensor_to_str(param))

    return resStr


def log_model(name, model):
    print("")
    print("{} : ".format(name))

    for param in model.parameters():
        print("    {}".format(tensor_to_str(param)))

    print("")


def log_tensor(name, tensor):
    print("{} : {}".format(name, tensor_to_str(tensor)))
