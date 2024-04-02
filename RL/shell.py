import os
import re
from tqdm import tqdm
pattern = re.compile(r'([0-9]+)')   # re.I 表示忽略大小写
cmd = 'ps aux | grep multiprocessing-fork'
textlist = os.popen(cmd).readlines()
# textlist = textlist[::-1]
x = []
y = []
num = 0
for i in range(len(textlist)):
    pid = textlist[i].split(' ')
    while '' in pid:
        pid.remove('')
    pid = pid[1]
    m = pattern.match(pid)
    if m is not None:
        # print(m.group())
        # print('pid:',pid)
        cmd = 'lsof -w -p {}'.format(pid)
        textlist1 = os.popen(cmd).readlines()
        for j in range(len(textlist1)):
            if "/home/jiashuncheng/code/Trasient/RL/train_dir/" in textlist1[j]:
                    x = textlist1[j]
                    print(x)
                    # c = input("是否终止？")
                    # if c == '1':
                    # xxx
                    print('正在kill:{}'.format(pid))
                    os.popen('kill -9 {}'.format(pid))
                    num+=1
print('共终止了{}个程序。'.format(num))