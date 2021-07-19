def serial_read(device_path, delay=0.1, file_path=None):
    from threading import Thread, Event
    from utils.hm310p import HMReader
    from serial import Serial
    from time import time, sleep

    def _wait_input(event):
        input()
        event.set()

    reader = HMReader(device_path)
    reader.setPower(1)
    event = Event()
    thread = Thread(target=_wait_input, args=[event])
    thread.start()

    print('Press any key to exit')

    if file_path is not None:
        with open(file_path, 'w') as f:
            f.write('serial_data\n')

    while True:
        timestamp = time()
        ret = reader.read()

        if file_path is not None:
            with open(file_path, 'a') as f:
                f.write('%f %.8f %.8f %.8f\n' % (timestamp,
                    ret['Current'], ret['Voltage'], ret['Power']))

        if not event.is_set():
            print('\x1b[1K\r%.6fA %.6fV %.6fW' % (ret['Current'],
                ret['Voltage'], ret['Power']), end='')
        else:
            break

        sleep(delay)

    thread.join()
    reader.close()


def serial_plot(device_path, delay=0.1, file_path=None):
    import matplotlib.animation as animation
    from matplotlib import axes
    import matplotlib.pyplot as plt
    from utils.hm310p import HMReader
    from time import time, sleep

    reader = HMReader(device_path)
    reader.setPower(1)
    time_list, power_list, voltage_list, current_list = list(), list(), list(), list()

    if file_path is not None:
        with open(file_path, 'w') as f:
            f.write('serial_data\n')

    def anim(i):
        timestamp = time()
        ret = reader.read()

        if file_path is not None:
            with open(file_path, 'a') as f:
                f.write('%f %.8f %.8f %.8f\n' % (timestamp,
                    ret['Current'], ret['Voltage'], ret['Power']))

        time_list.append(timestamp)

        power_list.append(ret['Power'])
        ax[0].clear()
        ax[0].relim()
        ax[0].autoscale()
        ax[0].plot(time_list, power_list)
        ax[0].set_ylim(bottom=0)
        ax[0].set_xlabel('t')
        ax[0].set_ylabel('W')

        voltage_list.append(ret['Voltage'])
        ax[1].clear()
        ax[1].relim()
        ax[1].autoscale()
        ax[1].plot(time_list, voltage_list)
        ax[1].set_ylim(bottom=0)
        ax[1].set_xlabel('t')
        ax[1].set_ylabel('V')

        current_list.append(ret['Current'])
        ax[2].clear()
        ax[2].relim()
        ax[2].autoscale()
        ax[2].plot(time_list, current_list)
        ax[2].set_ylim(bottom=0)
        ax[2].set_xlabel('t')
        ax[2].set_ylabel('A')

    fig, ax = plt.subplots(3)
    ani = animation.FuncAnimation(fig, anim, interval=(delay*1000))
    plt.show()

    reader.close()


def plot(file_paths, batch_on=True):
    import numpy as np
    from matplotlib import axes
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from cycler import cycler

    serial_data, inf_data, mem_data, accs = list(), list(), list(), list()
    for path in file_paths:
        with open(path, 'r') as f:
            lines = f.readlines()
            if 'serial_data' in lines[0]:
                time_list, current_list, voltage_list, power_list = list(), list(), list(), list()
                for line in lines[1:]:
                    line_args = line.split()
                    if len(line_args) == 4:
                        try:
                            time = float(line_args[0])
                            current = float(line_args[1])
                            voltage = float(line_args[2])
                            power = float(line_args[3])
                            time_list.append(time)
                            current_list.append(current)
                            voltage_list.append(voltage)
                            power_list.append(power)
                        except ValueError:
                            pass
                serial_data.append((time_list, current_list, voltage_list, power_list))
            elif 'label_data' in lines[0]:
                data_list = list()
                acc_label = lines.pop(-1).split()
                accs.append(acc_label[1])
                for line in lines[1:]:
                    line_args = line.split()
                    if len(line_args) == 2 and (batch_on or 'batch' not in line_args[1]):# and not 'top' in line_args[1]: # and ('init' in line_args[1] or 'test' in line_args[1] or 'top' in line_args[1]):
                        try:
                            time = float(line_args[0])
                            data_list.append((time, line_args[1]))
                        except ValueError:
                            pass
                #data_list.append((float(acc_label[0]), 'eval_end'))
                inf_data.append(data_list)
            elif 'mem_data' in lines[0]:
                time_list, mem_list, swap_list = list(), list(), list()
                for line in lines[1:]:
                    line_args = line.split()
                    try:
                        time = float(line_args[0])
                        mem_free = int(line_args[1]) - int(line_args[2])
                        swap_free = int(line_args[15]) - int(line_args[16])
                        time_list.append(time)
                        mem_list.append(mem_free)
                        swap_list.append(swap_free)
                    except ValueError:
                        pass
                mem_data.append((time_list, mem_list, swap_list))

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    fig, ax = plt.subplots(figsize=(4,3))
    plt.subplots_adjust(top=0.715,
        bottom=0.342,
        left=0.099,
        right=0.947,
        hspace=0.868,
        wspace=0.2)

    #plt.xlabel('Seconds', fontsize=7)
    #lines, names = [Line2D([0], [0], color=color, lw=4) for color in colors[3:]], accs
    lines, names = list(), list()
    if len(serial_data) > 0:
        #ax.set_ylabel('Power consumption in watts', fontsize=7)
        lines.insert(0, Line2D([0], [0], color=colors[0], lw=4))
        names.insert(0, 'Power in Watts')
        if len(mem_data) > 0:
            ax_mem = ax.twinx()
            lines = [Line2D([0], [0], color=color, lw=4) for color in colors[1:3]] + lines
            names = ['Allocated memory in GiB', 'Allocated swap in GiB'] + names
    elif len(mem_data) > 0:
        ax_mem = ax
        lines = [Line2D([0], [0], color=color, lw=4) for color in colors[0:2]] + lines
        names = ['Allocated memory in GiB', 'Allocated swap in GiB'] + names

    #t0 = min([time_list[0][0] for time_list in serial_data] + [data[0][0] for data in inf_data] + [data[0][0] for data in mem_data])
    t0 = min([data[0][0] for data in inf_data])
    x_min = 0
    x_max = max([data[-1][0] for data in inf_data]) - t0
    for data_list, color in zip(inf_data, colors[3:]):
        up = 1
        for (time, label) in data_list:
            time = time - t0
            plt.axvline(time, color=color)
            if up:
                plt.text(time, -0.2, label, rotation=45, transform=ax.get_xaxis_transform(), horizontalalignment='right', rotation_mode='anchor', fontsize=7)
                up = 0
            else:
                plt.text(time, 1.05, label, rotation=45, transform=ax.get_xaxis_transform(), fontsize=7)
                up = 1
    for time_list, mem_list, swap_list in mem_data:
        #ax_mem.set_ylabel('Memory in kB', fontsize=7)
        line = ax_mem.plot(np.array(time_list) - t0, np.array(mem_list) / (2**20), color=colors[0])
        ax_mem.plot(np.array(time_list) - t0, np.array(swap_list) / (2**20), color=colors[1])
    for (time_list, current_list, voltage_list, power_list) in serial_data:
        ax.plot(np.array(time_list) - t0, power_list, color=colors[0])

    ax.set_xlabel('Time in seconds', fontsize=7)
    ax.tick_params(axis='both', labelsize=7)
    #plt.legend([Line2D([0], [0], color=color, lw=4) for color in colors], accs, loc='upper center', bbox_to_anchor=(0, -.14))
    plt.legend(lines, names, loc='upper center', bbox_to_anchor=(0.5, -.35), fontsize=7)
    #plt.legend(lines, names, loc='lower right', fontsize=7)
    plt.ylim(bottom=0)
    plt.xlim(left=x_min-0.02*(x_max-x_min), right=x_max+0.05*(x_max-x_min))
    plt.ylim(bottom=0, top=4)
    plt.tight_layout()
    plt.savefig('fig.pdf')
    plt.show()

  
def calc_pow_averages_from_file(files, detailed=False):
    # Read power file into lists
    ret_list = list()
    pow_time_list, pow_list = None, None

    # Process label files
    for file_name in files:
        # Read label file into list
        with open(file_name, 'r') as f:
            lines = f.read().split('\n')
            if lines[0] == 'label_data':
                label_time_list, label_list = list(), list()

                for line in lines[1:]:
                    tokens = line.split()
                    if len(tokens) == 2:
                        label_time_list.append(float(tokens[0]))
                        label_list.append(tokens[1])
                ret_list.append((file_name, _process_label_data(label_time_list, label_list, pow_time_list, pow_list, detailed)))
                        
            elif lines[0] == 'serial_data':
                pow_time_list, pow_list = list(), list()

                for line in lines[1:]:
                    tokens = line.split()
                    if len(tokens) == 4:
                        pow_time_list.append(float(tokens[0]))
                        pow_list.append(float(tokens[3]))

    return ret_list


def _process_label_data(label_time_list, label_list, pow_time_list=None, pow_list=None, detailed=False):
    # Calculate averages and durations
    # Timestamps: [test_start, init_inf_start, init_inf_end, ..., test_end, accuracy]
    ret, inf_durations = dict(), list()

    for time_start, time_end in zip(label_time_list[3:-2:2], label_time_list[4:-2:2]):
        duration = time_end - time_start
        if duration < 0:
            print('Error while parsing')
            continue
        inf_durations.append(duration)

    init_duration = label_time_list[2] - label_time_list[1]
    test_duration = label_time_list[-2] - label_time_list[0]
    inf_duration = sum(inf_durations)
    inf_average_duration = inf_duration / len(inf_durations)

    if not detailed:
        ret['test_duration'] = '%.3f' % (test_duration)
        ret['start_duration'] = '%.3f' % (label_time_list[1] - label_time_list[0])
        ret['init_duration'] = '%.3f' % (init_duration)
        ret['inf_durations'] = '%.3f/ %.3f' % (inf_duration, inf_average_duration)
        ret['eval_duration'] = '%.3f' % (label_time_list[-1] - label_time_list[-2])
        ret['postproc_duration'] = '%.3f' % (label_time_list[-2] - label_time_list[-3])
    
    
        if pow_time_list is not None:
            inf_averages = list()
            for time_start, time_end in zip(label_time_list[3:-2:2], label_time_list[4:-2:2]):
                average = _calc_pow_average(pow_time_list, pow_list, time_start, time_end)
                if average < 0:
                    #print('Error while parsing')
                    continue
                inf_averages.append(average)
    
            init_power = _calc_pow_average(pow_time_list, pow_list, label_time_list[1], label_time_list[2])
            inf_average_power = sum(inf_averages) / len(inf_averages)
            test_average_power = _calc_pow_average(pow_time_list, pow_list, label_time_list[0], label_time_list[-2])
    
            ret['test_power'] = '%.3f/ %.3f' % (test_average_power * (test_duration / 60), test_average_power)
            ret['init_power'] = '%.3f/ %.3f' % (init_power * (init_duration / 60), init_power)
            ret['inf_power'] = '%.3f/ %.3f' % (inf_average_power * (inf_duration / 60), inf_average_power)

    else:
        ret['test_duration'] = test_duration
        ret['start_duration'] = label_time_list[1] - label_time_list[0]
        ret['init_duration'] = init_duration
        ret['inf_durations'] = inf_duration
        ret['inf_durations_avg'] = inf_average_duration
        ret['postproc_duration'] = label_time_list[-2] - label_time_list[-3]
        ret['eval_duration'] = label_time_list[-1] - label_time_list[-2]
    
    
        if pow_time_list is not None:
            inf_averages = list()
            for time_start, time_end in zip(label_time_list[3:-2:2], label_time_list[4:-2:2]):
                average = _calc_pow_average(pow_time_list, pow_list, time_start, time_end)
                if average < 0:
                    #print('Error while parsing')
                    continue
                inf_averages.append(average)
    
            init_power = _calc_pow_average(pow_time_list, pow_list, label_time_list[1], label_time_list[2])
            inf_average_power = sum(inf_averages) / len(inf_averages)
            test_average_power = _calc_pow_average(pow_time_list, pow_list, label_time_list[0], label_time_list[-2])
    
            ret['test_power'] = test_average_power * (test_duration / 60)
            ret['test_power_avg'] = test_average_power
            ret['init_power'] = init_power * (init_duration / 60)
            ret['init_power_avg'] = init_power
            ret['inf_power'] = inf_average_power * (inf_duration / 60)
            ret['inf_power_avg'] = inf_average_power



    #ret['test_wattminutes'] = test_average_power * (test_duration / 60)
    #ret['test_average_power'] = test_average_power

    #ret['init_wattminutes'] = init_power * (init_duration / 60)
    #ret['init_average_power'] = init_power

    #ret['inf_wattminutes'] = inf_average_power * (inf_duration / 60)
    #ret['inf_average_power'] = inf_average_power

    #ret['inf_average_duration'] = inf_average_duration

    ret['accuracy'] = label_list[-1]

    return ret


def _calc_pow_average(time_list, pow_list, time_start, time_end):
    start_index = 0
    for i, time in enumerate(time_list):
        if time >= time_start:
            start_index = i
            break

    pow_sum = 0.0
    pow_list = pow_list[start_index:]
    time_list = time_list[start_index:]
    end_index = 0
    for i, time in enumerate(time_list):
        pow_sum += pow_list[i]
        if time > time_end:
            end_index = i+1
            break

    return pow_sum / end_index if end_index > 0 else -1


def calc_idle(file_name):
    with open(file_name, 'r') as f:
        power, cnt = 0.0, 0
        for line in f.readlines():
            line = line.split()
            if len(line) == 4:
                power += float(line[3])
                cnt += 1
    
        return power/cnt
