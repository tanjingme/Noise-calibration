# from __future__ import with_statement
# from fabric.api import local,cd,run,env,task,sudo,put,prefix,parallel,get,settings
# from fabric.contrib.console import confirm
# from contextlib import contextmanager as _contextmanager
# from fabric.contrib import files
# import glog
# 远程服务器的连接参数
import argparse
from colour_demosaicing import (
    demosaicing_CFA_Bayer_bilinear,
    demosaicing_CFA_Bayer_Malvar2004,
    demosaicing_CFA_Bayer_Menon2007,
    mosaicing_CFA_Bayer)

import cv2
import numpy as np


DEFAULT_EXPOSURE=",".join(list(map(str,[1/4000, 1/3200, 1/2500, 1/2000, 1/1600, 1/1250, 1/1000, 1/800, 1/640, 1/500, 1/400, 1/320, 1/250, 1/200, 1/160, 1/125, 1/100, 1/80, 1/60, 1/50, 1/40, 1/30, 1/25, 1/20, 1/15, 1/13, 1/10, 1/8, 1/6, 1/5, 1/4, 1/3, 0.4, 0.5, 0.6, 0.8,1.0,1.2,1.4,1.6,1.8,2.0])))
def parse_float_list(arg):

    try:
        values = list(map(eval,arg.split(',')))

        return values
    except ValueError:
        raise argparse.ArgumentTypeError("Invalid integer list format. Example: 1,2,3")

def download(connection,host,user,save_name,folder,exp,iso,seq_num=1):
    try:
        if not os.path.exists(os.path.join("temp_images",save_name)+".jpg"):
            result = connection.run('./save_tan {} {} {} {}'.format(save_name,seq_num,exp,iso))
            os.system(f'scp {user}@{host}:/home/{user}/{save_name}.bin {os.path.join(folder,save_name)+".CR3"}')
            show_img=rawpy.imread(os.path.join(folder,save_name)+".CR3").raw_image.astype("float32")/4095
            show_img=np.clip(demosaicing_CFA_Bayer_bilinear(show_img**(1.0/2.2),pattern='RGGB')*255*1,0,255).astype("uint8")
            cv2.imwrite(os.path.join("temp_images",save_name)+".jpg",show_img[:,:,[2,1,0]])
            result = connection.run('rm {}.bin'.format(save_name))
        return False
    except Exception as e:
        print("invalid data --- download again")
        return True

import time
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Example script using argparse')
    # Add arguments
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose mode')
    parser.add_argument('-iso',default="3100",type=parse_float_list, help='specify iso')
    parser.add_argument('-out',default="images",type=str, help='Output dir path')
    parser.add_argument('-si',default=0,type=int, help='specify iso')
    parser.add_argument('-exps',default=DEFAULT_EXPOSURE,type=parse_float_list, help='specify iso')

    # Parse the command-line arguments
    args = parser.parse_args()
    exps=args.exps
    # assert(args.si%(len(args.exps)*len(args.iso))==0)
    print(f"total_image:{len(args.exps)*len(args.iso)*2}")

    print(args.exps)
    print(args.iso)
    host = '192.168.198.198'
    user = 'root'
    password = 'your_password'
    import os
    from fabric import Connection
    import rawpy
    connection= Connection(host=host, user=user)
    result = connection.run('ls -l')
    connection.run('./print_reg')
    time.sleep(1)
    connection.run('fpga_tools reg w 0x204 1')
    time.sleep(1)
    connection.run('/usr/sbin/i2ctransfer -f -y 0 w2@0x1a 0x9c 0x28 w8 0x00 0x01 0x00 0x01 0x00 0x01 0x00 0x01')
    time.sleep(1)
    connection.run('/usr/sbin/i2ctransfer -f -y 0 w2@0x1a 0xb4 0x88 w2 0x00 0x01')
    time.sleep(1)
    connection.run('/usr/sbin/i2ctransfer -f -y 0 w2@0x1a 0xb4 0x5e w16 0x00 0xe0 0x00 0xe0 0x00 0xe0 0x00 0xe0 0x00 0xe0 0x00 0xe0 0x00 0xe0 0x00 0xe0')
    time.sleep(1)
    connection.run('/usr/sbin/i2ctransfer -f -y 0 w2@0x1a 0xb4 0x6e w16 0x00 0xe0 0x00 0xe0 0x00 0xe0 0x00 0xe0 0x00 0xe0 0x00 0xe0 0x00 0xe0 0x00 0xe0')
    time.sleep(1)
    os.makedirs('temp_images',exist_ok=True)
    os.makedirs('images',exist_ok=True)
    folder=args.out
    count=0
    for j,iso in enumerate(args.iso):
        print(f"{iso}=================={count}")
        for i,exp in enumerate(exps):
            
            
            save_name=f"_MG_0{args.si+count}"
            print(save_name,iso,exp,'==========================')
            unfinished=True
            while unfinished:
                unfinished=download(connection,host,user,save_name,folder,exp,iso)

            save_name=f"_MG_0{args.si+count+1}"
            unfinished=True
            while unfinished:
                unfinished=download(connection,host,user,save_name,folder,exp,iso)

            count=count+2

