import numpy as np
import matplotlib.pyplot as plt
import sys

if __name__ == '__main__':
    y_list = list()
    for arg in sys.argv[1:]:
        try:
            arg = arg.split(',')
            if len(arg) == 2:
                pow_idle, pow_inf = float(arg[0]), float(arg[1])
                y_list.append((pow_idle, pow_inf))
                continue
        except ValueError:
            pass

        print("Error: Illegal argument")
        exit(1)


    x = np.arange(0, 1, 0.01)
    y_list = list(map(lambda p: 60 * (x * p[0] + (1 - x) * p[1]), y_list))

    for y in y_list:
        plt.plot(x, y)

    plt.xlabel('Time spent on idle')
    plt.ylabel('Power consumption in one hour')
    plt.show()
