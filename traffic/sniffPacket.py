from scapy.all import *
from scapy.layers.inet import TCP, IP, UDP
import tensorflow as tf
import numpy as np
import os


def add_flow(flows, num):
    key = 'flow'+str(num)
    value = 'flowlist'+str(num)
    value = []
    flows[key] = value
    return value


def sniff_packet(num):
    print("Start to sniff packets...")
    # dpkt = sniff(iface="enp3s0", filter="tcp and (src net 10.108.126.3 or dst net 10.108.126.3)", count=num)
    dpkt = sniff(iface="ens33", filter="ip", count=num)
    print("Sniffing packets done")
    return dpkt


def output(flow):
    all = flow[5:]
    li = np.reshape(all, (-1, 5))
    return li


def test_data(n):
    dpkt = sniff_packet(n)
    num = 1
    test_images = []
    flows = {}
    add_flow(flows, num)
    flows['flow1'] = [dpkt[0][Ether].src, dpkt[0][Ether].dst, dpkt[0][Ether].sport, dpkt[0][Ether].dport,
                      dpkt[0][Ether].proto]
    for packet in dpkt:
        IsGet = False
        srcIP = packet[Ether].src
        dstIP = packet[Ether].dst
        srcPort = packet[Ether].sport
        dstPort = packet[Ether].dport
        protocol = packet[Ether].proto
        timetolive = packet[IP].ttl
        if protocol == 6:
            window = packet[Ether].window
        else:
            window = 0
        length = packet[Ether].len
        # if packet[IP].src == "10.108.126.3":
        #     dir = 1
        # elif packet[IP].dst == "10.108.126.3":
        #     dir = 0
        print(srcIP, dstIP, srcPort, dstPort, protocol)
        for i in range(1, num + 1):
            flow = flows['flow' + str(i)]
            # print("i:" + str(i))
            IPs = (srcIP == flow[0] and dstIP == flow[1]) or (srcIP == flow[1] and dstIP == flow[0])
            Ports = (srcPort == flow[2] and dstPort == flow[3]) or (srcPort == flow[3] and dstPort == flow[2])
            Proto = (protocol == flow[4])
            if IPs and Ports and Proto:
                # flow.append(srcPort, dstPort, dir, timetolive, window, length)
                flow.append(srcPort)
                flow.append(dstPort)
                # flow.append(dir)
                flow.append(timetolive)
                flow.append(window)
                flow.append(length)
                # print("+1")
                # print(flows)
                IsGet = True
                if len(flow) == 30:
                    test_images = output(flow)
                    flow.append(packet)
                    return test_images, flow
                break
        if not IsGet:
            num = num + 1
            add_flow(flows, num)
            flows['flow' + str(num)] = [srcIP, dstIP, srcPort, dstPort, protocol, srcPort, dstPort, dir, timetolive, window, length]
            # print(flows)
    return test_images, []


def voteresult(predictionlist):
    label = []
    for row in predictionlist:
        label.append(row.index(max(row)))
    a = [0, 0, 0, 0, 0, 0, 0]
    for i in label:
        a[i] = label.count(i)
    # print(label)
    return a.index(max(a))


if __name__ == '__main__':
    # test_images = [[43550.0, 57163.0, .0, 1471.0, 109.0, .0]]
    # test_labels = [[1, 0, 0, 0, 0, 0, 0],
    #                [1, 0, 0, 0, 0, 0, 0],
    #                [1, 0, 0, 0, 0, 0, 0],
    #                [1, 0, 0, 0, 0, 0, 0]]
    # with tf.Session() as sess:
    test_images, flow = test_data(50)
    if len(flow) <= 1:
        print("do not get enough packet to classify")
    else:
        packet = flow[30]
        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())
        new_saver = tf.train.import_meta_graph('model/my-model5.meta')
        new_saver.restore(sess, 'model/my-model5')
        y_predict = tf.get_collection('pred_network')[0]
        graph = tf.get_default_graph()

        input_x = graph.get_operation_by_name('input_x').outputs[0]
        keep_prob = graph.get_operation_by_name('keep_prob').outputs[0]

        test_predict = y_predict.eval(feed_dict={input_x: test_images, keep_prob: 0.5})
        # sess.run(y_predict, feed_dict={input_x: test_images, keep_prob: 0.5})
        result = voteresult(test_predict.tolist())
        priority = result * 4
        packet[IP].tos = priority
        send(packet)
        print(test_images)
        print(test_predict)
        print("预测结果是："+str(result))
        print(packet.show())



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


