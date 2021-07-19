#!/usr/bin/python3

from time import time, sleep
from threading import Thread, Event
import argparse

def _waitinput(event):
    input('Press any key to exit')
    event.set()
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser('Memory measuring tool')
    parser.add_argument('file', help='File to save data to')
    parser.add_argument('-i', '--interval', type=float, help='Time interval for measuring in seconds, defaults to 0.1', default=0.1)
    args = parser.parse_args()
   
    event = Event()
    Thread(target=_waitinput, args=[event]).start()

    with open(args.file, 'w') as f:
        f.write('mem_data\n')
 
    with open(args.file, 'a') as fw:
        while not event.is_set():
            with open('/proc/meminfo', 'r') as fr:
                fw.write('%f %s\n' % (time(), ' '.join([x.split()[1] for x in fr.read().splitlines()])))
                sleep(args.interval)
