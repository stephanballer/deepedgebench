import numpy as np
import matplotlib.pyplot as plt
import sys

if __name__ == '__main__':
    y_list, name_list = list(), list()
    for arg in sys.argv[1:]:
        try:
            arg = arg.split(',')
            if len(arg) == 4:
                pow_idle, pow_inf, time_inf = float(arg[1]), float(arg[2]), float(arg[3])
                name_list.append(arg[0])
                y_list.append((pow_idle, pow_inf, time_inf))
                continue
        except ValueError:
            pass

        print("Error: Illegal argument")
        exit(1)


#    x = np.arange(0, 1, 0.01)
#    y_list = list(map(lambda p: 60 * (x * p[0] + (1 - x) * p[1]), y_list))
#   t_max / t_inf = fps
    t_max = 60
    x_in = np.arange(0, 100000, 1)
    for i,y in enumerate(y_list):
        y_out = list()
        for x in x_in: 
            if t_max - (x * y[2]) > .0:
                y_out.append(((t_max - (x * y[2])) * y[0] + (x * y[2]) * y[1])/60.0)
            else:
                break

        plt.plot(x_in[:len(y_out)], y_out)
        plt.plot(x_in[len(y_out)-1], y_out[-1], '.', label='_nolegend_', color='black')
        #plt.text(x_in[len(y_out)-1], y_out[-1], '%d, %.3fWm' % (x_in[len(y_out)-1], y_out[-1]), horizontalalignment='center')
        name_list[i] += ' (%d, %.3fWm)' % (x_in[len(y_out)-1], y_out[-1])

    plt.legend(name_list)
#    plt.xscale('log')
#    plt.yscale('log')
    plt.xlabel('Number of inferences in one minute')
    plt.ylabel('Power consumption in wattminutes')
    plt.show()
