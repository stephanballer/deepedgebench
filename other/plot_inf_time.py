import numpy as np
import matplotlib.pyplot as plt
import sys

def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])

if __name__ == '__main__':
    all_models = ['MobileNetV2', 'MobileNetV2 Lite', 'MobileNetV2 Quant. Lite', 'MobileNetV1 Quant. Lite']

    #fig, axs = plt.subplots(1, 4, figsize=(8.7,3))
    #plt.subplots_adjust(top=0.998,
    #    bottom=0.21,
    #    left=0.067,
    #    right=0.998,
    #    hspace=0.868,
    #    wspace=0.2)
    fig, axs = plt.subplots(2, 2, figsize=(5.78,4))
    plt.subplots_adjust(top=0.998,
        bottom=0.21,
        left=0.092,
        right=0.995,
        hspace=0.2,
        wspace=0.2)

    for model_num in range(len(all_models)):
        model_list = [all_models[model_num]]
        #dev_values = dict()
        #for arg in sys.argv[1:]: #    try:
        #        arg = arg.split(',')
        #        if len(arg) == (len(model_list)+1):
        #            model_values = list()
        #            for i in range(1, len(model_list)+1):
        #                model_values.append(float(arg[i]))
        #            dev_values[arg[0]] = model_values
        #            continue
        #    except ValueError:
        #        pass
    
        #    print("Error: Illegal argument")
        #    exit(1)
        dev_values = {'Tinker Edge R' : ([1652.010,938.587,584.854,22.055],[760.686,760.591,96.715,23.969]),
                'Raspberry Pi 4' :        ([1037.727 ,965.071,640.035,22.204],[1037.727,965.071,640.035,22.204]),
                'Coral Dev Board':      ([1389.365 ,1780.581,1081.275,37.563],[float('nan'),float('nan'),20.788,5.423]),
                'Jetson Nano':          ([685.490  ,905.093,657.725,22.957],[102.142,float('nan'),float('nan'),float('nan')])
                }
    
    
    #    x = np.arange(0, 1, 0.01)
    #    y_list = list(map(lambda p: 60 * (x * p[0] + (1 - x) * p[1]), y_list))
    #   t_max / t_inf = fps
        x = np.arange(len(model_list))
        width = 0.35
        shift = width/len(model_list)
    
        #colors = ['#76448A', '#922B21', '#1E8449', '#B7950B', '#d43d51']
        #colors_opt = ['#D2B4DE', '#E6B0AA','#A9DFBF','#F9E79F','#d43d51']
        dev_opt_labels = list()
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        colors_opt = colors[:2] + colors[3:4] + [colors[2]]
        colors = list(map(lambda c: lighten_color(c), colors_opt))
        for i,dev in enumerate(dev_values.items()):
            #plt.plot(dev[1], model_list, label=dev[0], marker='|', linestyle='None', markersize=20, mew=5)
            if dev[1][0][model_num] > dev[1][1][model_num]:
                axs[int(model_num/2)][model_num%2].bar(x-width/2+i*shift+0.5*shift, dev[1][0][model_num], width/len(model_list), label=dev[0], color=colors[i])
                if 'Rasp' not in dev[0]:
                    opt_label = dev[0]+' (optimized)'
                    axs[int(model_num/2)][model_num%2].bar(x-width/2+i*shift+0.5*shift, dev[1][1][model_num], width/len(model_list), label=opt_label, color=colors_opt[i])
                    dev_opt_labels.append(opt_label)
            else:
                if 'Rasp' not in dev[0]:
                    opt_label = dev[0]+' (optimized)'
                    axs[int(model_num/2)][model_num%2].bar(x-width/2+i*shift+0.5*shift, dev[1][1][model_num], width/len(model_list), label=opt_label, color=colors_opt[i])
                    dev_opt_labels.append(opt_label)
                axs[int(model_num/2)][model_num%2].bar(x-width/2+i*shift+0.5*shift, dev[1][0][model_num], width/len(model_list), label=dev[0], color=colors[i])
    
        axs[int(model_num/2)][model_num%2].set_xticks(np.arange(.0, 1.4, .35))
        #axs[int(model_num/2)][model_num%2].set_xscale('log')
        axs[int(model_num/2)][model_num%2].tick_params(axis='y', labelsize=7)
        axs[int(model_num/2)][model_num%2].set_xticklabels([])
        axs[int(model_num/2)][model_num%2].set_xlabel(all_models[model_num], fontsize=7)
        #ax.set_yticklabels(model_list)

    fig.subplots_adjust(hspace=0.2)
    plt.legend(loc='upper center', bbox_to_anchor=(-0.2, -.2), ncol=3, fontsize=7)
    #axs[0].legend(list(dev_values.keys()), loc='upper center', bbox_to_anchor=(1.5, 0), ncol=2, fontsize=8)
    #axs[3].legend(dev_opt_labels, loc='upper center', bbox_to_anchor=(-0.5, 0), ncol=2, fontsize=8)
    #fig = legend.figure
    #fig.canvas.draw()
    #bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    #fig.savefig('fig.svg', dpi="figure", bbox_inches=bbox)
    axs[0][0].set_ylabel('Time in seconds', fontsize=7)
    axs[1][0].set_ylabel('Time in seconds', fontsize=7)
    plt.show()
