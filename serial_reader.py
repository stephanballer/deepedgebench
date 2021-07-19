#!/usr/bin/python3

from utils.serial_utils import *
from utils.acs_utils import *
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Power measuring tool to use with 10 bit serial UART output values of a microcontroller with ACS712 5A")
    subparsers = parser.add_subparsers(dest='cmd')

    monitorparser = subparsers.add_parser('monitor', help='Monitor given device for serial input (HM310P)')
    monitorparser.add_argument('device_path', type=str, help='Path to serial device from which analog input data is read')
    monitorparser.add_argument('-i', '--interval',  type=float, default=0.1, help='Writing interval of average current that is measured')
    monitorparser.add_argument('-f', '--file', help='Write data to file with given path')
    monitorparser.add_argument('-l', '--live', action='store_true', default=False, help='Plot live graph instead of command line output')

    acsparser = subparsers.add_parser('acs_monitor', help='Monitor given device for serial input (ACS712 + Arduino Nano)')
    acsparser.add_argument('-c', '--calibrate',  type=float, default=-1.0, help='Calibrate for give number of seconds, default is 20')
    acsparser.add_argument('-b', '--base_value', type=float, default=512.0, help='Base value, use instead of calibrating')
    acsparser.add_argument('device_path', type=str, help='Path to serial device from which analog input data is read')
    acsparser.add_argument('-i', '--interval',  type=float, default=0.1, help='Writing interval of average current that is measured')
    acsparser.add_argument('-f', '--file', help='Write data to file with given path')
    acsparser.add_argument('-l', '--live', action='store_true', default=False, help='Plot live graph instead of command line output')

    plotparser = subparsers.add_parser('plot', help='Create plot from file paths')
    plotparser.add_argument('file_paths', nargs='+', help='path to data files')
    plotparser.add_argument('-eb', '--exclude_batches', action='store_true', help='Exclude batch timestamps from plot')

    evalparser = subparsers.add_parser('eval', help='Evaluate data from files')
    evalparser.add_argument('files', nargs='*', help='Files to process (power data file + label files to process)')
    evalparser.add_argument('-d', '--detailed', action='store_true', default=False, help='Show data with more than 3 decimals')

    evalidleparser = subparsers.add_parser('eval_power', help='Evaluate average power from power files')
    evalidleparser.add_argument('files', nargs='*')

    args = parser.parse_args()


    if args.cmd == 'plot':
        plot(args.file_paths, not args.exclude_batches)
    elif args.cmd == 'monitor':
        if args.live:
            serial_plot(args.device_path, args.interval, args.file)
        else:
            serial_read(args.device_path, args.interval, args.file)
    elif args.cmd == 'acs_monitor':
        acs_read(args.device_path, args.interval, args.file, args.calibrate, args.base_value)
    elif args.cmd == 'eval':
        for file_name, ret in calc_pow_averages_from_file(args.files, args.detailed):
            print('results["%s"] = %s' % (file_name.split('/')[-1].split('.')[-2], str(ret)))
    elif args.cmd == 'eval_power':
        for file_path in args.files:
            print('results["%s"] = %s' % (file_path.split('/')[-1].split('.')[-2], str(calc_idle(file_path))))
