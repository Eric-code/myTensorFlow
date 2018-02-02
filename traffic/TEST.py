from multiprocessing import Process, Queue
import os ,time, random, dill
from scapy.all import *

while True:
    dpkt = sniff(iface="enp3s0", filter="tcp", count=5)
    for packet in dpkt:
        print(packet.show())
