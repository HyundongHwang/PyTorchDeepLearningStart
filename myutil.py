g_oldLogType = "none"  # none, single, multi

def log(name, value):
    global g_oldLogType
    valueStr = "{}".format(value)

    if "\n" in valueStr:
        if g_oldLogType is "multi":
            print("")
        g_oldLogType = "multi"
        print("{} : ".format(name))
        print("    {}".format(valueStr))
        print("")
    else:
        g_oldLogType = "single"
        print("{} : {}", name, valueStr)
