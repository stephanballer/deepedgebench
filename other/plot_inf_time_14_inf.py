import numpy as np
import matplotlib.pyplot as plt
import sys
import colorsys
import matplotlib.colors as mc
import matplotlib as mpl
mpl.rcParams['hatch.linewidth'] = 0.06  # previous pdf hatch linewidth
mpl.rcParams['hatch.linewidth'] = 0.4  # previous svg hatch linewidth

pow_ts_tinker_tf_mn2 = {'test_duration': 1295.560152053833, 'init_duration': 0.5608601570129395, 'inf_durations': 1289.6158649921417, 'inf_durations_avg': 257.92317299842836, 'postproc_duration': 0.022544145584106445, 'eval_duration': 0.03786492347717285, 'test_power': 145.22985466852796, 'test_power_avg': 6.725887073863633, 'init_power': 0.05967552070617676, 'init_power_avg': 6.384, 'inf_power': 144.62601653762894, 'inf_power_avg': 6.728795161271232, 'accuracy': 'top_1_0.200__top_5_1.000'}
pow_ts_tinker_tflite_mn2 = {'test_duration': 913.1326739788055, 'init_duration': 0.28401803970336914, 'inf_durations': 912.0914459228516, 'inf_durations_avg': 182.4182891845703, 'postproc_duration': 0.0034661293029785156, 'eval_duration': 0.01620197296142578, 'test_power': 109.65366490897719, 'test_power_avg': 7.2051083944580645, 'init_power': 0.03192362766265869, 'init_power_avg': 6.744, 'inf_power': 109.53785404489847, 'inf_power_avg': 7.205715251549259, 'accuracy': 'top_1_0.200__top_5_1.000'}
pow_ts_tinker_tflite_mn2q = {'test_duration': 552.7306718826294, 'init_duration': 0.14018511772155762, 'inf_durations': 551.8821485042572, 'inf_durations_avg': 110.37642970085145, 'postproc_duration': 0.0016939640045166016, 'eval_duration': 0.012527942657470703, 'test_power': 64.09057336870994, 'test_power_avg': 6.957157613535082, 'init_power': 0.0150839186668396, 'init_power_avg': 6.456, 'inf_power': 63.998987265274145, 'inf_power_avg': 6.9578971639573295, 'accuracy': 'top_1_0.400__top_5_0.800'}
pow_ts_tinker_tflite_mn1q = {'test_duration': 19.27366304397583, 'init_duration': 0.004914999008178711, 'inf_durations': 18.581462144851685, 'inf_durations_avg': 3.7162924289703367, 'postproc_duration': 0.0005929470062255859, 'eval_duration': 0.012755155563354492, 'test_power': 2.188895086009682, 'test_power_avg': 6.814153846153835, 'init_power': 0.0005288538932800293, 'init_power_avg': 6.456, 'inf_power': 2.1146663963052044, 'inf_power_avg': 6.828310000000002, 'accuracy': 'top_1_0.000__top_5_0.600'}
pow_ts_tinker_rknn_mn2 = {'test_duration': 827.8831899166107, 'init_duration': 31.09830904006958, 'inf_durations': 794.591227054596, 'inf_durations_avg': 158.91824541091918, 'postproc_duration': 0.5486049652099609, 'eval_duration': 0.04585599899291992, 'test_power': 73.05667435542244, 'test_power_avg': 5.294708860759534, 'init_power': 2.749836878559119, 'init_power_avg': 5.305440000000012, 'inf_power': 70.08588130656376, 'inf_power_avg': 5.2922216294553825, 'accuracy': 'top_1_0.400__top_5_0.800'}
pow_ts_tinker_rknnlite_mn2 = {'test_duration': 825.564001083374, 'init_duration': 30.38224697113037, 'inf_durations': 793.2436368465424, 'inf_durations_avg': 158.64872736930846, 'postproc_duration': 0.5539281368255615, 'eval_duration': 0.04253888130187988, 'test_power': 72.83892230468486, 'test_power_avg': 5.293757155769841, 'init_power': 2.689792196483151, 'init_power_avg': 5.311902439024398, 'inf_power': 69.95315313613708, 'inf_power_avg': 5.291172841743444, 'accuracy': 'top_1_0.400__top_5_0.800'}
pow_ts_tinker_rknnlite_mn2q = {'test_duration': 131.33697700500488, 'init_duration': 26.535141944885254, 'inf_durations': 102.97416067123413, 'inf_durations_avg': 20.594832134246825, 'postproc_duration': 0.46535611152648926, 'eval_duration': 0.02555704116821289, 'test_power': 12.160066152253197, 'test_power_avg': 5.555206049149348, 'init_power': 2.355681795733066, 'init_power_avg': 5.3265555555555615, 'inf_power': 9.618659198978456, 'inf_power_avg': 5.60450843373493, 'accuracy': 'top_1_0.400__top_5_0.800'}
pow_ts_tinker_rknnlite_mn1q = {'test_duration': 36.418959856033325, 'init_duration': 5.5771260261535645, 'inf_durations': 29.421570301055908, 'inf_durations_avg': 5.884314060211182, 'postproc_duration': 0.11166691780090332, 'eval_duration': 0.04498004913330078, 'test_power': 3.319981909362365, 'test_power_avg': 5.4696486486486435, 'init_power': 0.5798352025190987, 'init_power_avg': 6.237999999999999, 'inf_power': 2.5857088215942383, 'inf_power_avg': 5.273088, 'accuracy': 'top_1_0.000__top_5_0.600'}
ts_tinker_tf_mn2 = {'test_duration': 2030.9558980464935, 'init_duration': 0.513221025466919, 'inf_durations': 1652.1005728244781, 'inf_durations_avg': 0.3304201145648956, 'postproc_duration': 0.022764205932617188, 'eval_duration': 18.784770011901855, 'test_power': 203.85181573563563, 'test_power_avg': 6.022340985298017, 'init_power': 0.054735022366046895, 'init_power_avg': 6.398999999999999, 'inf_power': 167.4264603409294, 'inf_power_avg': 6.080493999999976, 'accuracy': 'top_1_0.701__top_5_0.891'}
ts_tinker_tflite_mn2 = {'test_duration': 1042.582757949829, 'init_duration': 0.193342924118042, 'inf_durations': 938.5869574546814, 'inf_durations_avg': 0.18771739149093628, 'postproc_duration': 0.004600048065185547, 'eval_duration': 30.031326055526733, 'test_power': 123.59049000980211, 'test_power_avg': 7.112557103064014, 'init_power': 0.021925087594985963, 'init_power_avg': 6.804, 'inf_power': 111.16926149093393, 'inf_power_avg': 7.106593199999901, 'accuracy': 'top_1_0.701__top_5_0.891'}
ts_tinker_tflite_mn2q = {'test_duration': 679.9951369762421, 'init_duration': 0.11466407775878906, 'inf_durations': 584.8536744117737, 'inf_durations_avg': 0.11697073488235474, 'postproc_duration': 0.0018520355224609375, 'eval_duration': 15.755990982055664, 'test_power': 76.22851332453897, 'test_power_avg': 6.726093395035759, 'init_power': 0.012704779815673828, 'init_power_avg': 6.648, 'inf_power': 65.57618017804006, 'inf_power_avg': 6.72744480000004, 'accuracy': 'top_1_0.727__top_5_0.904'}
ts_tinker_tflite_mn1q = {'test_duration': 116.04415607452393, 'init_duration': 0.004789829254150391, 'inf_durations': 22.055476665496826, 'inf_durations_avg': 0.004411095333099365, 'postproc_duration': 0.00061798095703125, 'eval_duration': 15.612122058868408, 'test_power': 11.975756906890824, 'test_power_avg': 6.191999999999977, 'init_power': 0.0005163435935974121, 'init_power_avg': 6.468, 'inf_power': 2.2765892391083846, 'inf_power_avg': 6.193262400000191, 'accuracy': 'top_1_0.361__top_5_0.611'}
ts_tinker_rknn_mn2 = {'test_duration': 1077.6782200336456, 'init_duration': 30.356828927993774, 'inf_durations': 760.6855266094208, 'inf_durations_avg': 0.15213710532188415, 'postproc_duration': 0.5684561729431152, 'eval_duration': 14.487085819244385, 'test_power': 95.29653109488635, 'test_power_avg': 5.305657810839556, 'init_power': 2.6816711999362925, 'init_power_avg': 5.300299065420571, 'inf_power': 67.22025861542159, 'inf_power_avg': 5.302080000000022, 'accuracy': 'top_1_0.734__top_5_0.907'}
ts_tinker_rknnlite_mn2 = {'test_duration': 1078.047439813614, 'init_duration': 30.36374306678772, 'inf_durations': 760.5905125141144, 'inf_durations_avg': 0.15211810250282287, 'postproc_duration': 0.5675568580627441, 'eval_duration': 14.459927082061768, 'test_power': 95.29106318146484, 'test_power_avg': 5.303536356318786, 'init_power': 2.691249219596279, 'init_power_avg': 5.318018691588794, 'inf_power': 67.17944604220428, 'inf_power_avg': 5.2995228000000285, 'accuracy': 'top_1_0.734__top_5_0.908'}
ts_tinker_rknnlite_mn2q = {'test_duration': 331.41845893859863, 'init_duration': 26.419857025146484, 'inf_durations': 96.71494579315186, 'inf_durations_avg': 0.01934298915863037, 'postproc_duration': 0.4636080265045166, 'eval_duration': 9.93514108657837, 'test_power': 29.918340670997164, 'test_power_avg': 5.41641659311561, 'init_power': 2.3452217867561047, 'init_power_avg': 5.326043478260878, 'inf_power': 8.715171392415787, 'inf_power_avg': 5.406716400000021, 'accuracy': 'top_1_0.726__top_5_0.904'}
ts_tinker_rknnlite_mn1q = {'test_duration': 131.7984230518341, 'init_duration': 5.505250930786133, 'inf_durations': 23.96910572052002, 'inf_durations_avg': 0.004793821144104004, 'postproc_duration': 0.10934019088745117, 'eval_duration': 10.427947998046875, 'test_power': 12.961063096145656, 'test_power_avg': 5.9004028103044694, 'init_power': 0.5684722111129761, 'init_power_avg': 6.1956, 'inf_power': 2.3527623556074655, 'inf_power_avg': 5.889487199999937, 'accuracy': 'top_1_0.364__top_5_0.613'}
pow_ts_jetson_tf_mn2 = {'test_duration': 312.16963291168213, 'init_duration': 33.854166984558105, 'inf_durations': 248.94160056114197, 'inf_durations_avg': 49.78832011222839, 'postproc_duration': 0.45811891555786133, 'eval_duration': 0.09723305702209473, 'test_power': 33.21255154224194, 'test_power_avg': 6.383558432470267, 'init_power': 1.7532552247964424, 'init_power_avg': 3.1073076923076934, 'inf_power': 30.204082679645815, 'inf_power_avg': 7.279799586303566, 'accuracy': 'top_1_0.200__top_5_1.000'}
pow_ts_jetson_tflite_mn2 = {'test_duration': 913.8495218753815, 'init_duration': 0.2634718418121338, 'inf_durations': 911.5340890884399, 'inf_durations_avg': 182.306817817688, 'postproc_duration': 0.0033299922943115234, 'eval_duration': 0.020183086395263672, 'test_power': 52.39056917698152, 'test_power_avg': 3.439772167487713, 'init_power': 0.013400470621056029, 'init_power_avg': 3.051666666666667, 'inf_power': 52.291140237654815, 'inf_power_avg': 3.441964981690204, 'accuracy': 'top_1_0.200__top_5_1.000'}
pow_ts_jetson_tflite_mn2q = {'test_duration': 657.3272860050201, 'init_duration': 0.14420008659362793, 'inf_durations': 656.301317691803, 'inf_durations_avg': 131.2602635383606, 'postproc_duration': 0.0018219947814941406, 'eval_duration': 0.011218070983886719, 'test_power': 33.533313263618034, 'test_power_avg': 3.0608782544921103, 'init_power': 0.007222021003564199, 'init_power_avg': 3.005, 'inf_power': 33.4835513254462, 'inf_power_avg': 3.0611138898706254, 'accuracy': 'top_1_0.400__top_5_0.800'}
pow_ts_jetson_tflite_mn1q = {'test_duration': 22.604615926742554, 'init_duration': 0.0055010318756103516, 'inf_durations': 21.781056880950928, 'inf_durations_avg': 4.356211376190186, 'postproc_duration': 0.00046706199645996094, 'eval_duration': 0.011363983154296875, 'test_power': 1.122128783460159, 'test_power_avg': 2.978494623655914, 'init_power': 0.00025304746627807616, 'init_power_avg': 2.76, 'inf_power': 1.0847136159515753, 'inf_power_avg': 2.988046783625731, 'accuracy': 'top_1_0.000__top_5_0.600'}
pow_ts_jetson_tftrt_mn2 = {'test_duration': 304.5912058353424, 'init_duration': 32.27567100524902, 'inf_durations': 247.85086822509766, 'inf_durations_avg': 49.57017364501953, 'postproc_duration': 1.660585880279541, 'eval_duration': 0.10586309432983398, 'test_power': 33.71244474871847, 'test_power_avg': 6.640857142857157, 'init_power': 1.6075427466891699, 'init_power_avg': 2.988398437499999, 'inf_power': 30.64452015427135, 'inf_power_avg': 7.418457810631528, 'accuracy': 'top_1_0.200__top_5_1.000'}
pow_ts_jetson_onnxtrt_mn2 = {'test_duration': 172.5846140384674, 'init_duration': 54.44504690170288, 'inf_durations': 115.4796838760376, 'inf_durations_avg': 23.09593677520752, 'postproc_duration': 0.011764049530029297, 'eval_duration': 0.019798994064331055, 'test_power': 17.722759179323027, 'test_power_avg': 6.1614157014157005, 'init_power': 3.9399510046162582, 'init_power_avg': 4.341938775510205, 'inf_power': 13.629537253217668, 'inf_power_avg': 7.0815247127876, 'accuracy': 'top_1_0.200__top_5_1.000'}
ts_jetson_tf_mn2 = {'test_duration': 476.45284700393677, 'init_duration': 14.050250053405762, 'inf_durations': 103.87977981567383, 'inf_durations_avg': 0.020775955963134765, 'postproc_duration': 0.09998798370361328, 'eval_duration': 15.688064098358154, 'test_power': 43.61244587431755, 'test_power_avg': 5.492142126789373, 'init_power': 0.821440575526712, 'init_power_avg': 3.507868852459018, 'inf_power': 9.835253583459307, 'inf_power_avg': 5.680751499999996, 'accuracy': 'top_1_0.701__top_5_0.891'}
ts_jetson_tflite_mn2 = {'test_duration': 1085.1459650993347, 'init_duration': 0.28120899200439453, 'inf_durations': 905.0925908088684, 'inf_durations_avg': 0.1810185181617737, 'postproc_duration': 0.0036149024963378906, 'eval_duration': 27.04475998878479, 'test_power': 61.59883238043582, 'test_power_avg': 3.4059288443171076, 'init_power': 0.0162515363295873, 'init_power_avg': 3.4675, 'inf_power': 51.438802888595404, 'inf_power_avg': 3.409958500000002, 'accuracy': 'top_1_0.701__top_5_0.891'}
ts_jetson_tflite_mn2q = {'test_duration': 804.1733038425446, 'init_duration': 0.15091419219970703, 'inf_durations': 657.7254908084869, 'inf_durations_avg': 0.13154509816169738, 'postproc_duration': 0.0013208389282226562, 'eval_duration': 14.018028974533081, 'test_power': 40.45887649214872, 'test_power_avg': 3.018668461051312, 'init_power': 0.006552191178003946, 'init_power_avg': 2.605, 'inf_power': 33.13803541516874, 'inf_power_avg': 3.022966500000016, 'accuracy': 'top_1_0.729__top_5_0.905'}
ts_jetson_tflite_mn1q = {'test_duration': 146.87875890731812, 'init_duration': 0.005515098571777344, 'inf_durations': 22.956600427627563, 'inf_durations_avg': 0.004591320085525512, 'postproc_duration': 0.00043487548828125, 'eval_duration': 14.946563005447388, 'test_power': 7.195202694209031, 'test_power_avg': 2.9392416225749587, 'init_power': 0.00025231575965881353, 'init_power_avg': 2.745, 'inf_power': 1.1246394549343828, 'inf_power_avg': 2.939388499999975, 'accuracy': 'top_1_0.360__top_5_0.611'}
ts_jetson_tftrt_mn2 = {'test_duration': 474.80743193626404, 'init_duration': 17.626795053482056, 'inf_durations': 103.1424651145935, 'inf_durations_avg': 0.020628493022918703, 'postproc_duration': 0.394542932510376, 'eval_duration': 16.884501934051514, 'test_power': 41.96372098535574, 'test_power_avg': 5.302830347144451, 'init_power': 0.9466126850611168, 'init_power_avg': 3.2221830985915494, 'inf_power': 9.381762989015641, 'inf_power_avg': 5.45755599999998, 'accuracy': 'top_1_0.701__top_5_0.891'}
ts_jetson_onnxtrt_mn2 = {'test_duration': 350.67201685905457, 'init_duration': 53.67205190658569, 'inf_durations': 118.73664164543152, 'inf_durations_avg': 0.023747328329086305, 'postproc_duration': 0.010486841201782227, 'eval_duration': 28.228612184524536, 'test_power': 27.579206437020567, 'test_power_avg': 4.718803630363035, 'init_power': 3.777753031961414, 'init_power_avg': 4.223151041666668, 'inf_power': 9.573680005438339, 'inf_power_avg': 4.8377719999999815, 'accuracy': 'top_1_0.701__top_5_0.891'}
pow_ts_jetson_cpu_tffg_mn2 = {'test_duration': 683.713240146637, 'init_duration': 0.32793307304382324, 'inf_durations': 675.8237481117249, 'inf_durations_avg': 135.16474962234497, 'postproc_duration': 0.010226011276245117, 'eval_duration': 0.01556086540222168, 'test_power': 62.12801281804446, 'test_power_avg': 5.4521114265436585, 'init_power': 0.022053499162197114, 'init_power_avg': 4.035, 'inf_power': 61.76110420500688, 'inf_power_avg': 5.483184428860593, 'accuracy': 'top_1_0.200__top_5_1.000'}
pow_ts_jetson_cpu_tf_mn2 = {'test_duration': 635.4972438812256, 'init_duration': 1.1951799392700195, 'inf_durations': 622.6285533905029, 'inf_durations_avg': 124.52571067810058, 'postproc_duration': 0.014275074005126953, 'eval_duration': 0.01241612434387207, 'test_power': 56.95908479034047, 'test_power_avg': 5.377749660326091, 'init_power': 0.07971186206075878, 'init_power_avg': 4.001666666666666, 'inf_power': 56.333170086770366, 'inf_power_avg': 5.4285820764251795, 'accuracy': 'top_1_0.200__top_5_1.000'}
ts_jetson_cpu_tffg_mn2 = {'test_duration': 835.7073919773102, 'init_duration': 0.3276810646057129, 'inf_durations': 685.4895784854889, 'inf_durations_avg': 0.13709791569709778, 'postproc_duration': 0.010155916213989258, 'eval_duration': 12.52177906036377, 'test_power': 70.44862577734445, 'test_power_avg': 5.057891777933955, 'init_power': 0.025550020787451003, 'init_power_avg': 4.678333333333334, 'inf_power': 58.598591061714025, 'inf_power_avg': 5.129057500000009, 'accuracy': 'top_1_0.701__top_5_0.891'}
ts_jetson_cpu_tf_mn2 = {'test_duration': 789.3739221096039, 'init_duration': 1.2143230438232422, 'inf_durations': 631.5815806388855, 'inf_durations_avg': 0.1263163161277771, 'postproc_duration': 0.03876304626464844, 'eval_duration': 11.954427003860474, 'test_power': 65.66028525217104, 'test_power_avg': 4.990812344803113, 'init_power': 0.07761548121770223, 'init_power_avg': 3.8350000000000004, 'inf_power': 53.51688102414153, 'inf_power_avg': 5.084082499999993, 'accuracy': 'top_1_0.701__top_5_0.891'}
pow_ts_jetson_cpu_tffg_mn2 = {'test_duration': 683.713240146637, 'init_duration': 0.32793307304382324, 'inf_durations': 675.8237481117249, 'inf_durations_avg': 135.16474962234497, 'postproc_duration': 0.010226011276245117, 'eval_duration': 0.01556086540222168, 'test_power': 62.12801281804446, 'test_power_avg': 5.4521114265436585, 'init_power': 0.022053499162197114, 'init_power_avg': 4.035, 'inf_power': 61.76110420500688, 'inf_power_avg': 5.483184428860593, 'accuracy': 'top_1_0.200__top_5_1.000'}
pow_ts_jetson_cpu_tf_mn2 = {'test_duration': 635.4972438812256, 'init_duration': 1.1951799392700195, 'inf_durations': 622.6285533905029, 'inf_durations_avg': 124.52571067810058, 'postproc_duration': 0.014275074005126953, 'eval_duration': 0.01241612434387207, 'test_power': 56.95908479034047, 'test_power_avg': 5.377749660326091, 'init_power': 0.07971186206075878, 'init_power_avg': 4.001666666666666, 'inf_power': 56.333170086770366, 'inf_power_avg': 5.4285820764251795, 'accuracy': 'top_1_0.200__top_5_1.000'}
ts_jetson_cpu_tffg_mn2 = {'test_duration': 835.7073919773102, 'init_duration': 0.3276810646057129, 'inf_durations': 685.4895784854889, 'inf_durations_avg': 0.13709791569709778, 'postproc_duration': 0.010155916213989258, 'eval_duration': 12.52177906036377, 'test_power': 70.44862577734445, 'test_power_avg': 5.057891777933955, 'init_power': 0.025550020787451003, 'init_power_avg': 4.678333333333334, 'inf_power': 58.598591061714025, 'inf_power_avg': 5.129057500000009, 'accuracy': 'top_1_0.701__top_5_0.891'}
ts_jetson_cpu_tf_mn2 = {'test_duration': 789.3739221096039, 'init_duration': 1.2143230438232422, 'inf_durations': 631.5815806388855, 'inf_durations_avg': 0.1263163161277771, 'postproc_duration': 0.03876304626464844, 'eval_duration': 11.954427003860474, 'test_power': 65.66028525217104, 'test_power_avg': 4.990812344803113, 'init_power': 0.07761548121770223, 'init_power_avg': 3.8350000000000004, 'inf_power': 53.51688102414153, 'inf_power_avg': 5.084082499999993, 'accuracy': 'top_1_0.701__top_5_0.891'}
pow_ts_coral_tf_mn2 = {'test_duration': 1373.4692821502686, 'init_duration': 0.5327939987182617, 'inf_durations': 1364.4923939704895, 'inf_durations_avg': 272.8984787940979, 'postproc_duration': 0.01771402359008789, 'eval_duration': 0.025773048400878906, 'test_power': 113.12506460516231, 'test_power_avg': 4.941867986798653, 'init_power': 0.03825756907463073, 'init_power_avg': 4.308333333333333, 'inf_power': 112.62634713465097, 'inf_power_avg': 4.95245034559365, 'accuracy': 'top_1_0.200__top_5_1.000'}
pow_ts_coral_tflite_mn2 = {'test_duration': 1787.369106054306, 'init_duration': 0.8511569499969482, 'inf_durations': 1785.3694863319397, 'inf_durations_avg': 357.0738972663879, 'postproc_duration': 0.004762887954711914, 'eval_duration': 0.01769113540649414, 'test_power': 123.46917310207526, 'test_power_avg': 4.144723303670793, 'init_power': 0.05535357364813487, 'init_power_avg': 3.902, 'inf_power': 123.32992848236196, 'inf_power_avg': 4.1446858846818735, 'accuracy': 'top_1_0.200__top_5_1.000'}
pow_ts_coral_tflite_mn2q = {'test_duration': 1081.5719769001007, 'init_duration': 0.3077120780944824, 'inf_durations': 1080.2031881809235, 'inf_durations_avg': 216.0406376361847, 'postproc_duration': 0.0017838478088378906, 'eval_duration': 0.012827157974243164, 'test_power': 70.67807900598743, 'test_power_avg': 3.9208530092592584, 'init_power': 0.019932904614342585, 'init_power_avg': 3.8866666666666667, 'inf_power': 70.58593258542163, 'inf_power_avg': 3.92070306907477, 'accuracy': 'top_1_0.400__top_5_0.800'}
pow_ts_coral_tflite_mn1q = {'test_duration': 37.066051959991455, 'init_duration': 0.01544189453125, 'inf_durations': 36.07458853721619, 'inf_durations_avg': 7.214917707443237, 'postproc_duration': 0.0007250308990478516, 'eval_duration': 0.013453006744384766, 'test_power': 2.3713995976996034, 'test_power_avg': 3.8386601307189503, 'init_power': 0.0009432423909505208, 'init_power_avg': 3.665, 'inf_power': 2.3082905986157924, 'inf_power_avg': 3.8391965517241395, 'accuracy': 'top_1_0.000__top_5_0.600'}
pow_ts_coral_litetpu_mn2q = {'test_duration': 22.365796089172363, 'init_duration': 0.02426910400390625, 'inf_durations': 21.143038272857666, 'inf_durations_avg': 4.228607654571533, 'postproc_duration': 0.010149002075195312, 'eval_duration': 0.01831793785095215, 'test_power': 1.623650292283011, 'test_power_avg': 4.355714285714281, 'init_power': 0.0015491778055826823, 'init_power_avg': 3.83, 'inf_power': 1.5431582017959211, 'inf_power_avg': 4.3791952184382525, 'accuracy': 'top_1_0.400__top_5_0.800'}
pow_ts_coral_litetpu_mn1q = {'test_duration': 4.785645008087158, 'init_duration': 0.004271030426025391, 'inf_durations': 3.79010272026062, 'inf_durations_avg': 0.758020544052124, 'postproc_duration': 0.004606008529663086, 'eval_duration': 0.017632007598876953, 'test_power': 0.3222533707320689, 'test_power_avg': 4.040249999999999, 'init_power': 0.00027619330088297526, 'init_power_avg': 3.88, 'inf_power': 0.2543253677862882, 'inf_power_avg': 4.0261499999999995, 'accuracy': 'top_1_0.000__top_5_0.600'}
ts_coral_tf_mn2 = {'test_duration': 1695.0120770931244, 'init_duration': 0.5331380367279053, 'inf_durations': 1389.365165233612, 'inf_durations_avg': 0.2778730330467224, 'postproc_duration': 0.018177032470703125, 'eval_duration': 17.632218837738037, 'test_power': 134.38740993163566, 'test_power_avg': 4.757042563216582, 'init_power': 0.04219195185105007, 'init_power_avg': 4.748333333333334, 'inf_power': 112.25848236661128, 'inf_power_avg': 4.847903999999991, 'accuracy': 'top_1_0.701__top_5_0.891'}
ts_coral_tflite_mn2 = {'test_duration': 2106.0674378871918, 'init_duration': 0.846729040145874, 'inf_durations': 1780.5806047916412, 'inf_durations_avg': 0.35611612095832823, 'postproc_duration': 0.007898092269897461, 'eval_duration': 17.525213956832886, 'test_power': 145.17302507356024, 'test_power_avg': 4.135851182976303, 'init_power': 0.05630748116970062, 'init_power_avg': 3.9899999999999998, 'inf_power': 123.1101591147379, 'inf_power_avg': 4.148427499999999, 'accuracy': 'top_1_0.701__top_5_0.891'}
ts_coral_tflite_mn2q = {'test_duration': 1390.6286101341248, 'init_duration': 0.30602097511291504, 'inf_durations': 1081.2754497528076, 'inf_durations_avg': 0.21625508995056153, 'postproc_duration': 0.0019450187683105469, 'eval_duration': 13.634312868118286, 'test_power': 91.22316743698657, 'test_power_avg': 3.9359107142857437, 'init_power': 0.01945783366759618, 'init_power_avg': 3.815, 'inf_power': 70.93199388641932, 'inf_power_avg': 3.936018000000012, 'accuracy': 'top_1_0.729__top_5_0.905'}
ts_coral_tflite_mn1q = {'test_duration': 312.310916185379, 'init_duration': 0.015412092208862305, 'inf_durations': 37.56300973892212, 'inf_durations_avg': 0.007512601947784424, 'postproc_duration': 0.0006949901580810547, 'eval_duration': 14.151378870010376, 'test_power': 19.747276794839458, 'test_power_avg': 3.793772635814886, 'init_power': 0.0009722461501757305, 'init_power_avg': 3.785, 'inf_power': 2.3729964995193065, 'inf_power_avg': 3.7904254999999902, 'accuracy': 'top_1_0.360__top_5_0.611'}
ts_coral_litetpu_mn2q = {'test_duration': 316.0095109939575, 'init_duration': 0.02432084083557129, 'inf_durations': 20.788362503051758, 'inf_durations_avg': 0.004157672500610351, 'postproc_duration': 0.011523008346557617, 'eval_duration': 13.70735788345337, 'test_power': 21.03971854417469, 'test_power_avg': 3.9947630331753516, 'init_power': 0.0015666674971580507, 'init_power_avg': 3.865, 'inf_power': 1.3840694265088185, 'inf_power_avg': 3.99474300000003, 'accuracy': 'top_1_0.727__top_5_0.904'}
ts_coral_litetpu_mn1q = {'test_duration': 277.76090598106384, 'init_duration': 0.003371000289916992, 'inf_durations': 5.422756910324097, 'inf_durations_avg': 0.0010845513820648192, 'postproc_duration': 0.004438877105712891, 'eval_duration': 14.183290958404541, 'test_power': 18.30257572145104, 'test_power_avg': 3.953596491228065, 'init_power': 0.00024243110418319705, 'init_power_avg': 4.315, 'inf_power': 0.35706047977783156, 'inf_power_avg': 3.9506895000000077, 'accuracy': 'top_1_0.359__top_5_0.612'}
pow_ts_rpi_tf_mn2 = {'test_duration': 1070.0005881786346, 'init_duration': 0.6768851280212402, 'inf_durations': 1053.521279335022, 'inf_durations_avg': 210.7042558670044, 'postproc_duration': 0.008021116256713867, 'eval_duration': 0.18462800979614258, 'test_power': 83.59125324320445, 'test_power_avg': 4.687357418307271, 'init_power': 0.04473082554340363, 'init_power_avg': 3.9650000000000007, 'inf_power': 82.7192524293994, 'inf_power_avg': 4.711015565719457, 'accuracy': 'top_1_0.200__top_5_1.000'}
pow_ts_rpi_tflite_mn2 = {'test_duration': 999.1386709213257, 'init_duration': 0.6782200336456299, 'inf_durations': 997.4494638442993, 'inf_durations_avg': 199.48989276885987, 'postproc_duration': 0.0043048858642578125, 'eval_duration': 0.013432025909423828, 'test_power': 61.9687099411566, 'test_power_avg': 3.7213278843875, 'init_power': 0.03682169599334399, 'init_power_avg': 3.2575, 'inf_power': 61.86384740131758, 'inf_power_avg': 3.721322210925031, 'accuracy': 'top_1_0.200__top_5_1.000'}
pow_ts_rpi_tflite_mn2q = {'test_duration': 638.537880897522, 'init_duration': 0.2058849334716797, 'inf_durations': 637.3716368675232, 'inf_durations_avg': 127.47432737350464, 'postproc_duration': 0.001508951187133789, 'eval_duration': 0.009195089340209961, 'test_power': 37.71604143665562, 'test_power_avg': 3.543975312817059, 'init_power': 0.012009954452514649, 'init_power_avg': 3.5, 'inf_power': 37.63773839342263, 'inf_power_avg': 3.543088793068988, 'accuracy': 'top_1_0.400__top_5_0.800'}
pow_ts_rpi_tflite_mn1q = {'test_duration': 22.525207042694092, 'init_duration': 0.005947113037109375, 'inf_durations': 21.6739821434021, 'inf_durations_avg': 4.33479642868042, 'postproc_duration': 0.0005609989166259766, 'eval_duration': 0.00960683822631836, 'test_power': 1.3158103411020636, 'test_power_avg': 3.5049009900990145, 'init_power': 0.0003454281489054362, 'init_power_avg': 3.485, 'inf_power': 1.260389678720814, 'inf_power_avg': 3.4891318181818187, 'accuracy': 'top_1_0.000__top_5_0.600'}
ts_rpi_tf_mn2 = {'test_duration': 1166.1843619346619, 'init_duration': 0.4503591060638428, 'inf_durations': 1037.7269535064697, 'inf_durations_avg': 0.20754539070129394, 'postproc_duration': 0.008369922637939453, 'eval_duration': 14.905036926269531, 'test_power': 88.77057546523184, 'test_power_avg': 4.567231993299808, 'init_power': 0.031512627449300556, 'init_power_avg': 4.198333333333333, 'inf_power': 79.27763488570477, 'inf_power_avg': 4.583727999999984, 'accuracy': 'top_1_0.701__top_5_0.891'}
ts_rpi_tflite_mn2 = {'test_duration': 1070.071179151535, 'init_duration': 0.21861004829406738, 'inf_durations': 965.0710861682892, 'inf_durations_avg': 0.19301421723365783, 'postproc_duration': 0.007236003875732422, 'eval_duration': 22.64814281463623, 'test_power': 66.15324072806789, 'test_power_avg': 3.709280766566639, 'init_power': 0.013699563026428222, 'init_power_avg': 3.76, 'inf_power': 59.6470790023493, 'inf_power_avg': 3.7083534999999808, 'accuracy': 'top_1_0.701__top_5_0.891'}
ts_rpi_tflite_mn2q = {'test_duration': 735.6392729282379, 'init_duration': 0.13548898696899414, 'inf_durations': 640.035459280014, 'inf_durations_avg': 0.1280070918560028, 'postproc_duration': 0.0023148059844970703, 'eval_duration': 13.575858116149902, 'test_power': 43.29519195284602, 'test_power_avg': 3.5312300644723322, 'init_power': 0.007914814988772074, 'init_power_avg': 3.505, 'inf_power': 37.661414519775356, 'inf_power_avg': 3.530561999999932, 'accuracy': 'top_1_0.727__top_5_0.904'}
ts_rpi_tflite_mn1q = {'test_duration': 108.31880784034729, 'init_duration': 0.0057871341705322266, 'inf_durations': 22.20374059677124, 'inf_durations_avg': 0.004440748119354248, 'postproc_duration': 0.0005788803100585938, 'eval_duration': 14.237998008728027, 'test_power': 6.233138097917888, 'test_power_avg': 3.452662500000002, 'init_power': 0.0003330013453960418, 'init_power_avg': 3.4524999999999997, 'inf_power': 1.276048046940562, 'inf_power_avg': 3.448197499999938, 'accuracy': 'top_1_0.361__top_5_0.611'}
arduino = {'test_duration': 71516.99710416794, 'init_duration': 11.499000072479248, 'inf_durations': 57520.90205860138, 'inf_durations_avg': 11.506481708061889, 'postproc_duration': 0.19563508033752441, 'eval_duration': 2.3177528381347656, 'accuracy': 'top_1_0.188__top_5_0.423'}
arduino_init = 0.054051724137931016
inf_arduino = 0.06454545454545403
idle_coral = 3.08100985221677
idle_jetson = 1.3919730941704094
idle_rpi = 2.650000000000003
idle_tinker = 4.931999999999995
idle_nolan_arduino = 0.03639534883720928
idle_nolan_coral = 2.757317327766175
idle_nolan_jetson = 0.9028164556961911
idle_nolan_rpi = 2.099999999999996
idle_nolan_tinker = 4.775999999999984

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

def autolabel(rects, ax):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(round(height,1)),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 0),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize = 8.0)


if __name__ == '__main__':
    all_models = ['MobileNetV2', 'MobileNetV2 Lite', 'MobileNetV2 Quant. Lite', 'MobileNetV1 Quant. Lite']

    #fig, axs = plt.subplots(1, 4, figsize=(8.7,3))
    #plt.subplots_adjust(top=0.998,
    #    bottom=0.21,
    #    left=0.067,
    #    right=0.998,
    #    hspace=0.868,
    #    wspace=0.2)
    fig, axs = plt.subplots(1, 4, figsize=(8,3))
    for ax in axs:
        ax.set_axisbelow(True)
        ax.grid(axis="both", color="0.9", linestyle='-', linewidth=0.5)
    plt.subplots_adjust(top=0.998,
        bottom=0.315,
        left=0.102,
        right=0.973,
        hspace=0.868,
        wspace=0.2)

    for model_num, model in enumerate(all_models):
#        dev_values = {'Tinker Edge R' : ([1652.010,938.587,584.854,22.055],[760.686,760.591,96.715,23.969]),
#                'Raspberry Pi 4' :        ([1037.727 ,965.071,640.035,22.204],[1037.727,965.071,640.035,22.204]),
#                'Coral Dev Board':      ([1389.365 ,1780.581,1081.275,37.563],[float('nan'),float('nan'),20.788,5.423]),
#                'Jetson Nano':          ([685.490  ,905.093,657.725,22.957],[102.142,float('nan'),float('nan'),float('nan')])
#                }
        dev_values = {'Tinker Edge R' : ([ts_tinker_tf_mn2,       ts_tinker_tflite_mn2, ts_tinker_tflite_mn2q, ts_tinker_tflite_mn1q],
                    [ts_tinker_rknn_mn2, ts_tinker_rknnlite_mn2, ts_tinker_rknnlite_mn2q, ts_tinker_rknnlite_mn1q]),
                'Raspberry Pi 4' :      ([ts_rpi_tf_mn2,          ts_rpi_tflite_mn2,    ts_rpi_tflite_mn2q,    ts_rpi_tflite_mn1q],
                    [float('nan'),float('nan'),float('nan'),float('nan')]),
                'Coral Dev Board':      ([ts_coral_tf_mn2,        ts_coral_tflite_mn2,  ts_coral_tflite_mn2q,  ts_coral_tflite_mn1q],
                    [float('nan'),float('nan'),ts_coral_litetpu_mn2q,ts_coral_litetpu_mn1q]),
                'Jetson Nano':          ([ts_jetson_cpu_tffg_mn2, ts_jetson_tflite_mn2, ts_jetson_tflite_mn2q, ts_jetson_tflite_mn1q],
                    [ts_jetson_onnxtrt_mn2,float('nan'),float('nan'),float('nan')], [ts_jetson_tftrt_mn2,float('nan'),float('nan'),float('nan')])
                }
#        dev_values = {'Tinker Edge R' : ([pow_ts_tinker_tf_mn2,       pow_ts_tinker_tflite_mn2, pow_ts_tinker_tflite_mn2q, pow_ts_tinker_tflite_mn1q],
#                    [pow_ts_tinker_rknn_mn2, pow_ts_tinker_rknnlite_mn2, pow_ts_tinker_rknnlite_mn2q, pow_ts_tinker_rknnlite_mn1q]),
#                'Raspberry Pi 4' :      ([pow_ts_rpi_tf_mn2,          pow_ts_rpi_tflite_mn2,    pow_ts_rpi_tflite_mn2q,    pow_ts_rpi_tflite_mn1q],
#                    [float('nan'),float('nan'),float('nan'),float('nan')]),
#                'Coral Dev Board':      ([pow_ts_coral_tf_mn2,        pow_ts_coral_tflite_mn2,  pow_ts_coral_tflite_mn2q,  pow_ts_coral_tflite_mn1q],
#                    [float('nan'),float('nan'),pow_ts_coral_litetpu_mn2q,pow_ts_coral_litetpu_mn1q]),
#                'Jetson Nano':          ([pow_ts_jetson_cpu_tffg_mn2, pow_ts_jetson_tflite_mn2, pow_ts_jetson_tflite_mn2q, pow_ts_jetson_tflite_mn1q],
#                    [pow_ts_jetson_onnxtrt_mn2,float('nan'),float('nan'),float('nan')], [pow_ts_jetson_tftrt_mn2,float('nan'),float('nan'),float('nan')])
#                }
        for k, l in dev_values.items():
            for i in l:
                for ii, data in enumerate(i):
                    if type(data) != float:
                        i[ii] = data['inf_durations']

    #    x = np.arange(0, 1, 0.01)
    #    y_list = list(map(lambda p: 60 * (x * p[0] + (1 - x) * p[1]), y_list))
    #   t_max / t_inf = fps
        width = 0.4
    
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

        hatch = ['---', '+++', 'xxx', '\\\\\\', '***', 'ooo', 'OOO', '...']

        hatch1 = ['***', 'ooo', 'OOO', '...', '**', 'oo', 'OO', '..']
        colors_opt = colors[:2] + colors[3:4] + [colors[2]]
        colors = list(map(lambda c: lighten_color(c), colors_opt))


        for i,dev in enumerate(dev_values.items()):
            #plt.plot(dev[1], model_list, label=dev[0], marker='|', linestyle='None', markersize=20, mew=5)
            if dev[1][0][model_num] > dev[1][1][model_num]:
                opt_label = dev[0]
                #p1 = axs[model_num].bar(i*width, dev[1][0][model_num], width, label=opt_label, color=colors[i], hatch=hatch[i])
                p1 = axs[model_num].bar(i*width, dev[1][0][model_num], width, label=opt_label, color=colors[i])
                autolabel(p1,  axs[model_num])

                if 'Rasp' not in dev[0]:
                    opt_label = dev[0]+' (with AI unit)'
                    if 'Jetson' in dev[0]:
                        opt_label = dev[0]+' (with AI unit, TRT)'
                    #p1 = axs[model_num].bar(i*width, dev[1][1][model_num], width, label=opt_label, color=colors_opt[i], hatch=hatch1[i])
                    p1 = axs[model_num].bar(i*width, dev[1][1][model_num], width, label=opt_label, color=colors_opt[i])
                    autolabel(p1, axs[model_num])

                if 'Jetson' in dev[0]:
                    opt_label = dev[0]+' (with AI unit, TF-TRT)'
                    #p1 = axs[model_num].bar(i*width, dev[1][2][model_num], width, label=opt_label, color=lighten_color(colors[i]), hatch = '|||')
                    p1 = axs[model_num].bar(i*width, dev[1][2][model_num], width, label=opt_label, color=lighten_color(colors[i]))
                    #autolabel(p1, axs[model_num])

            else:
                if 'Rasp' not in dev[0]:
                    opt_label = dev[0]+' (with AI unit)'
                    if 'Jetson' in dev[0]:
                        opt_label = dev[0]+' (with AI unit, TRT)'

                    #ax.bar(x-3*width, aws, width=width , capsize=2, ecolor='blue', edgecolor='red', label='AWS', color = 'w', hatch = 'oo' ) 
                    #p1 = axs[model_num].bar(0-width/2+i*width+0.5*width, dev[1][1][model_num], width, label=opt_label, color=colors_opt[i], hatch=hatch1[i])
                    p1 = axs[model_num].bar(0-width/2+i*width+0.5*width, dev[1][1][model_num], width, label=opt_label, color=colors_opt[i])
                    autolabel(p1, axs[model_num])

                if 'Jetson' in dev[0]:
                    opt_label = dev[0]+' (with AI unit, TF-TRT)'
                    #p1 = axs[model_num].bar(i*width, dev[1][2][model_num], width, label=opt_label, color=lighten_color(colors[i]), hatch = '|||')
                    p1 = axs[model_num].bar(i*width, dev[1][2][model_num], width, label=opt_label, color=lighten_color(colors[i]))
                    autolabel(p1, axs[model_num])

                #p1 = axs[model_num].bar(i*width, dev[1][0][model_num], width, label=dev[0], color=colors[i], hatch=hatch[i])
                p1 = axs[model_num].bar(i*width, dev[1][0][model_num], width, label=dev[0], color=colors[i])
                autolabel(p1, axs[model_num])
    
        axs[model_num].set_xticks(np.arange(.0, 1.4, .35))
        #axs[model_num].set_xscale('log')
        axs[model_num].tick_params(axis='y', labelsize=8)
        axs[model_num].set_xticklabels([])
        axs[model_num].set_xlabel(model, fontsize=9)
        #ax.set_yticklabels(model_list)

    fig.subplots_adjust(wspace=0.28)
    plt.legend(loc='upper center', bbox_to_anchor=(-1.4, -.14), ncol=3, fontsize=9)
    #axs[0].legend(list(dev_values.keys()), loc='upper center', bbox_to_anchor=(1.5, 0), ncol=2, fontsize=8)
    #axs[3].legend(dev_opt_labels, loc='upper center', bbox_to_anchor=(-0.5, 0), ncol=2, fontsize=8)
    #fig = legend.figure
    #fig.canvas.draw()
    #bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    #fig.savefig('fig.svg', dpi="figure", bbox_inches=bbox)
    axs[0].set_ylim(axs[1].get_ylim())
    axs[2].set_ylim(axs[1].get_ylim())
    axs[0].set_ylabel('Duration in seconds', fontsize=9)
#    axs[1].set_ylabel('Duration in seconds', fontsize=9)
#    axs[2].set_ylabel('Duration in seconds', fontsize=9)
#    axs[3].set_ylabel('Duration in seconds', fontsize=9)
    plt.show()
    #fig.savefig("plot_inf_time_14_inf.pdf", format='pdf', dpi=300, bbox_inches='tight')
    fig.savefig("plot_inf_time_14_inf.svg", bbox_inches='tight')
