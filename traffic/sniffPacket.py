from scapy.all import *
from scapy.layers.inet import TCP, IP, UDP
import tensorflow as tf
import numpy as np
import os


def sniff_packet(num):
    print("start sniff packet...")
    dpkt = sniff(iface="enp3s0", filter="tcp and (src net 10.108.126.3 or dst net 10.108.126.3)", count=num)
    print("finish sniff packet...")
    li = []
    for packet in dpkt:
        p = [packet[Ether].sport, packet[Ether].dport]
        if packet[IP].src == "10.108.126.3":
            p.append(1)
        elif packet[IP].dst == "10.108.126.3":
            p.append(0)
        p.append(packet[IP].ttl)
        p.append(packet[Ether].window)
        p.append(packet[Ether].len)
        li.append(p)
    return li


if __name__ == '__main__':
    # test_images = [[43550.0, 57163.0, .0, 1471.0, 109.0, .0]]
    # test_labels = [[1, 0, 0, 0, 0, 0, 0],
    #                [1, 0, 0, 0, 0, 0, 0],
    #                [1, 0, 0, 0, 0, 0, 0],
    #                [1, 0, 0, 0, 0, 0, 0]]
    # with tf.Session() as sess:
    test_images = sniff_packet(4)
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    new_saver = tf.train.import_meta_graph('model/my-model.meta')
    new_saver.restore(sess, 'model/my-model')
    y_predict = tf.get_collection('pred_network')[0]
    graph = tf.get_default_graph()

    input_x = graph.get_operation_by_name('input_x').outputs[0]
    keep_prob = graph.get_operation_by_name('keep_prob').outputs[0]

    test_predict = y_predict.eval(feed_dict={input_x: test_images, keep_prob: 0.5})
    # sess.run(y_predict, feed_dict={input_x: test_images, keep_prob: 0.5})
    print(test_images)
    print(test_predict)


# dpkt = sniff(iface="enp3s0", filter="tcp and (src net 10.108.126.3 or dst net 10.108.126.3)", count=5)
# for packet in dpkt:
#     # print(packet.show())
#     print(packet[Ether].sport)
#     print(packet[Ether].dport)
#     print(packet[IP].src)
#     print(packet[IP].dst)
#     if packet[IP].src == "10.108.126.3":
#         print(1)
#     elif packet[IP].dst == "10.108.126.3":
#         print(0)
#     print(packet[IP].ttl)
#     print(packet[Ether].window)
#     print(packet[Ether].len)
#     print()
# pkts = []
# count = 0
# pcapnum = 0
# filename = ''
#
#
# def test_dump_file(dump_file):
#     print("Testing the dump file...")
#
#     if os.path.exists(dump_file):
#         print("dump fie %s found." % dump_file)
#         pkts = sniff(offline=dump_file)
#         count = 0
#         while (count <= 2):
#             print("----Dumping pkt:%s----" % dump_file)
#             print(hexdump(pkts[count]))
#             count += 1
#     else:
#         print("dump fie %s not found." % dump_file)
#
#
# def write_cap(x):
#     global pkts
#     global count
#     global pcapnum
#     global filename
#     pkts.append(x)
#     count += 1
#     if count == 3:         # 每3个TCP操作封为一个包（为了检测正确性，使用时尽量增多）</span>
#         pcapnum += 1
#         pname = "pcap%d.pcap" % pcapnum
#         wrpcap(pname, pkts)
#         filename = "./pcap%d.pcap" % pcapnum
#         test_dump_file(filename)
#         pkts = []
#         count = 0
#
#
# if __name__ == '__main__':
#     print("Start packet capturing and dumping ...")
#     sniff(filter="dst net 127.0.0.1 and tcp", prn=write_cap)  # BPF过滤规则


