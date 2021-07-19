#!/usr/bin/python3


""" Circuit:
                                  ____________
                                 |            |
      Uin _______________________|  ACS712 5A |________ Uin device
                |                |____________|
               _|_                 |      |
              |   |              Imes     |
          R1  |   |              to A0    |
              |___|                       |
                |_________ Umes           |
               _|_         zu A1          |
              |   |                       |
          R2  |   |                       |
              |_ _|           GND         |
                |            to AGND      |
    GND ________|______________|__________|__________ GND device

    Arduino out: "A0 A1\n" in mV
    I = (Umes / max_val) * 5000mV - 2500mV) / 185mV
    R1: 10kO  R2: 2.2kO
    U = (Umes / max_val) * 5.0V * Rfac
    Rfac = (R1 + R2) / R2
"""

from serial import Serial
from time import time, sleep

res_factor = (9.93 + 1.98) / 1.98
max_val = 1023.0
tolerance = 512

def acs_read(device_path, delay=0.1, file_path=None, calibrate=-1.0, zero_val=512.0):
    ser = Serial(device_path, 9600)
    rec = str()

    if calibrate >= 0.0:
        print("Calibrating for %d seconds..." % (calibrate))

        rec += ser.read().decode('utf-8', errors='ignore')
        sleep(calibrate)
        rec += ser.read(ser.inWaiting()).decode('utf-8', errors='ignore')
        num_list = list()
        for line in rec.split("\n")[1:-1]:
            nums = line.split()
            if len(nums) == 2:
                try:
                    num = float(nums[0])
                    if num >= 0.0 and num <= max_val:
                        num_list.append(num)
                except ValueError:
                    pass

        if len(num_list) > 0:
            zero_val = sum(num_list)/len(num_list)
        else:
            print('Error: no readable data received')
            exit()

        usr_inp = input("%.2f analog input value in relation to 1023 or %.3fV in relation to ~5V as base value. Press any key to continue or \'q\' to exit\n" % (zero_val, (zero_val/max_val) * 5.0))
        if len(usr_inp) > 0 and usr_inp[0] == "q":
            exit()

        rec = str()

    if file_path is not None:
        with open(file_path, 'w') as f:
            f.write('serial_data\n')

    while True:
        rec += ser.read().decode('utf-8', errors='ignore')
        sleep(delay)
        rec += ser.read(ser.inWaiting()).decode('utf-8', errors='ignore')
        
        timestamp = time()
        rec_list = rec.split('\n')
        rec = rec_list[-1]
        vol_list, cur_list = list(), list()
        for line in rec_list[:-1]:
            nums = line.split()
            if len(nums) == 2:
                try:
                    cur_num, vol_num = float(nums[0]), float(nums[1])

                    if cur_num >= zero_val - tolerance and cur_num <= max_val and vol_num >= 0 and vol_num <= max_val:
                        voltage = (vol_num / max_val) * 5.0 * res_factor
                        vol_list.append(voltage)
 
                        current = ((cur_num - zero_val)/max_val) * 5000.0 / 185.0
                        cur_list.append(current)
                except ValueError:
                    pass

        if (len(vol_list) > 0):
            voltage = sum(vol_list)/len(vol_list)
            current = sum(cur_list)/len(cur_list)
            if file_path is not None:
                with open(file_path, 'a') as f:
                    f.write('%f %.8f %.8f\n' % (timestamp, current, voltage))

            print('\x1b[1K\r%.6fA %.6fV %.6fW' % (current, voltage, current*voltage), end='')
    ser.close()
