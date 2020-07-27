import matplotlib.pyplot as plt  # 맷플롯립사용

g_oldLogType = "none"  # none, single, multi


def to_str(value):
    valueTypeStr = "{}".format(type(value))
    resStr = ""
    is_torch_nn_modules = False

    try:
        getattr(value, "parameters")
        is_torch_nn_modules = True
    except:
        is_torch_nn_modules = False

    if is_torch_nn_modules:
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


g_epoch_array = []
g_cost_array = []
g_accurcy_array = []

def log_epoch(epoch, nb_epoches, cost, accuracy=None, model=None):
    global g_epoch_array
    global g_cost_array
    global g_accurcy_array

    g_epoch_array.append(epoch)
    g_cost_array.append(cost.item())

    logStr = ""
    logStr += "{} \n".format("-" * 80)
    logStr += "epoch : {:4d}/{} \n".format(epoch, nb_epoches)
    logStr += "cost : {:.6f} \n".format(cost.item())

    if accuracy != None:
        g_accurcy_array.append(accuracy)
        logStr += "accuracy : {:2.2f} \n".format(accuracy)

    if model != None:
        logStr += "model : \n"
        for line in to_str(model).splitlines():
            logStr += "    {} \n".format(line)

    print(logStr)




def plt_init():
    global g_epoch_array
    global g_cost_array
    global g_accurcy_array
    g_epoch_array = []
    g_cost_array = []
    g_accurcy_array = []

def plt_show():
    plt.xlabel("epoch")
    plt.plot(g_epoch_array, g_cost_array, label="cost")

    if len(g_accurcy_array) > 0:
        plt.plot(g_epoch_array, g_accurcy_array, label="accurcy")

    plt.legend()
    plt.show()
