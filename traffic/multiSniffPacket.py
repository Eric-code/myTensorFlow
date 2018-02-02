from multiprocessing import Process, Queue
import os ,time, random, dill
from scapy.all import *
from scapy.layers.inet import TCP, IP, UDP
import tensorflow as tf
import numpy as np
import socket


def add_flow(flows, num):
    key = 'flow'+str(num)
    value = 'flowlist'+str(num)
    value = []
    flows[key] = value
    return value


def sniff_packet(num):
    print("Start to sniff packets...")
    dpkt = sniff(iface="enp3s0", filter="tcp and (src net 10.108.126.3 or dst net 10.108.126.3)", count=num)
    # dpkt = sniff(iface="ens33", filter="ip", count=num)
    print("Sniffing packets done")
    return dpkt


def output(flow):
    all = flow[5:]
    li = np.reshape(all, (-1, 5))
    return li


def put_model(packet, testdata):
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    new_saver = tf.train.import_meta_graph('model/my-model5.meta')
    new_saver.restore(sess, 'model/my-model5')
    y_predict = tf.get_collection('pred_network')[0]
    graph = tf.get_default_graph()

    input_x = graph.get_operation_by_name('input_x').outputs[0]
    keep_prob = graph.get_operation_by_name('keep_prob').outputs[0]

    test_predict = y_predict.eval(feed_dict={input_x: testdata, keep_prob: 0.5})
    result = vote_result(test_predict.tolist())
    priority = result * 4
    packet[IP].tos = priority
    send(packet)
    # print(testdata)
    # print(test_predict)
    print("预测结果是：" + str(result))
    return result


def vote_result(predictionlist):
    label = []
    for row in predictionlist:
        label.append(row.index(max(row)))
    a = [0, 0, 0, 0, 0, 0, 0]
    for i in label:
        a[i] = label.count(i)
    # print(label)
    return a.index(max(a))


# 写数据进程执行的代码:
def write(q):
    print('Process to write: %s' % os.getpid())
    count = 0
    while True:
        value = sniff(iface="enp3s0", filter="tcp", count=10)
        count = count + 10
        print(count)
        q.put(dill.dumps(value))


# 读数据进程执行的代码:
def read(q):
    print('Process to read: %s' % os.getpid())
    dpkt = sniff_packet(1)
    num = 1
    test_images = []
    flows = {}
    add_flow(flows, num)
    flows['flow1'] = [dpkt[0][IP].src, dpkt[0][IP].dst, dpkt[0][Ether].sport, dpkt[0][Ether].dport,
                      dpkt[0][Ether].proto]
    classifiedflows = []
    classifiedlabels = []
    IsFind = False  # 本次抓取中是否进行了流的分类
    while True:
        if q.qsize() == 0:
            continue
        else:
            dpkt = dill.loads(q.get(True))  # True表示在在队列满时采用阻塞模式
            for packet in dpkt:
                IsGet = False
                srcIP = packet[IP].src
                dstIP = packet[IP].dst
                srcPort = packet[Ether].sport
                dstPort = packet[Ether].dport
                protocol = packet[Ether].proto
                # 判断接收的包是不是属于已经被分类的流，如果是就不再进行分类
                fivetuple = [srcIP, dstIP, srcPort, dstPort, protocol]
                if fivetuple in classifiedflows:
                    continue
                timetolive = packet[IP].ttl
                if protocol == 6:
                    window = packet[Ether].window
                else:
                    window = 0
                length = packet[Ether].len
                print(srcIP, dstIP, srcPort, dstPort, protocol)
                for i in range(1, num + 1):
                    flow = flows['flow' + str(i)]
                    IPs = (srcIP == flow[0] and dstIP == flow[1]) or (srcIP == flow[1] and dstIP == flow[0])
                    Ports = (srcPort == flow[2] and dstPort == flow[3]) or (srcPort == flow[3] and dstPort == flow[2])
                    Proto = (protocol == flow[4])
                    if IPs and Ports and Proto:
                        flow.append(srcPort)
                        flow.append(dstPort)
                        # flow.append(dir)
                        flow.append(timetolive)
                        flow.append(window)
                        flow.append(length)
                        IsGet = True
                        if len(flow) == 30:
                            IsFind = True
                            test_images = output(flow)
                            result = put_model(packet, test_images)
                            classifiedflows.append([srcIP, dstIP, srcPort, dstPort, protocol])
                            classifiedflows.append([dstIP, srcIP, dstPort, srcPort, protocol])
                            classifiedlabels.append(result)
                            classifiedlabels.append(result)
                if not IsGet:
                    num = num + 1
                    add_flow(flows, num)
                    flows['flow' + str(num)] = [srcIP, dstIP, srcPort, dstPort, protocol, srcPort, dstPort, timetolive,
                                                window, length]
            if not IsFind:
                print("do not get enough packet to classify")


if __name__ == '__main__':
    # 父进程创建Queue，设置队列长度无限长，并传给各个子进程：
    q = Queue(maxsize=0)
    pw = Process(target=write, args=(q,))
    pr = Process(target=read, args=(q,))
    # 启动子进程pw，写入:
    pw.start()
    # 启动子进程pr，读取:
    pr.start()
    # time.sleep(30)
    # 等待pw结束:
    # pw.join()
    # pw.terminate()
    # print('pw is over')
    # pr进程里是死循环，无法等待其结束，只能强行终止:
    # time.sleep(5)
    # pr.terminate()
    # print('pr is over')
    q.close()

