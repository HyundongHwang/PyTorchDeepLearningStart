g_oldLogType = "none"  # none, single, multi


def to_str(value):
    valueTypeStr = "{}".format(type(value))
    resStr = ""

    if "torch.nn.modules.container." in valueTypeStr:
        resStr += "{} \n".format(value)
        for param in value.parameters():
            resStr += "{} {}\n".format(param.shape, param.data)
    elif "torch.Tensor" in valueTypeStr:
        resStr += "{} {}\n".format(value.shape, value.data)
    else:
        resStr += "{}".format(value)

    return resStr

def log(name, value):
    global g_oldLogType
    valueStr = to_str(value)

    if "\n" in valueStr:
        if g_oldLogType != "multi":
            print("")
        g_oldLogType = "multi"
        print("{} : ".format(name))
        for line in valueStr.splitlines():
            print("    {}".format(line))
        print("")
    else:
        g_oldLogType = "single"
        print("{} : {}".format(name, valueStr))