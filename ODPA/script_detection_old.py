#!/usr/bin/python
# -*- coding:utf-8 -*-
import subprocess,time,sys
import logging
from logging import handlers

class Logger(object):
    level_relations = {
        'debug':logging.DEBUG,
        'info':logging.INFO,
        'warning':logging.WARNING,
        'error':logging.ERROR,
        'crit':logging.CRITICAL
    }#日志级别关系映射

    def __init__(self,filename,level='info',when='D',backCount=3,fmt='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'):
        self.logger = logging.getLogger(filename)
        format_str = logging.Formatter(fmt)#设置日志格式
        self.logger.setLevel(self.level_relations.get(level))#设置日志级别
        sh = logging.StreamHandler()#往屏幕上输出
        sh.setFormatter(format_str) #设置屏幕上显示的格式
        th = handlers.TimedRotatingFileHandler(filename=filename,when=when,backupCount=backCount,encoding='utf-8')#往文件里写入#指定间隔时间自动生成文件的处理器
        #实例化TimedRotatingFileHandler
        #interval是时间间隔，backupCount是备份文件的个数，如果超过这个个数，就会自动删除，when是间隔的时间单位，单位有以下几种：
        # S 秒
        # M 分
        # H 小时
        # D 天
        # W 每星期（interval==0时代表星期一）
        # midnight 每天凌晨
        th.setFormatter(format_str)#设置文件里写入的格式
        self.logger.addHandler(sh) #把对象加到logger里
        self.logger.addHandler(th)

TIME = 1                        #程序状态检测间隔（单位：分钟）
CMD = "/home/jiashuncheng/code/Trasient/ODPA/old/model_torch.py" #需要执行程序的绝对路径，支持jar 如：D:\\calc.exe 或者D:\\test.jar

list_ = []
for x in range(1,11):
    for y in range(1,11):
        list_.append(['{:.1f}'.format(x*0.1), '{:.1f}'.format(y*0.1)])
        
def args(i):
    _args = ["--exp", "0.3_seed_1_old_auto_trasient_sparse_heirachical_con_{}_m_{}".format(list_[i][0], list_[i][1]), "--delay", "6", "--gpu", "1", "--num_iterations", "1000", "--seed", "1", "--mode", "train", "--max_delay", "6000", "--L2", "0.1", "--transient", "True", "--regions", "True", "--connection_prob", "{}".format(list_[i][0]), "--EI", "0.8", "--learnable_m", "False", "--m", "{}".format(list_[i][1]), "--noise_h", "True"]
    return _args

class Auto_Run():
    def __init__(self,sleep_time,cmd):
        self.log = Logger('all.log',level='debug')
        self.sleep_time = sleep_time
        self.cmd = cmd
        self.ext = (cmd[-3:]).lower()        #判断文件的后缀名，全部换成小写
        self.p = None             #self.p为subprocess.Popen()的返回值，初始化为None

        self.index = 0
        self.args = args(self.index)

        self.run()                           #启动时先执行一次程序
 
        try:
            while 1:
                time.sleep(sleep_time * 60)  #休息1分钟，判断程序状态
                self.poll_1 = self.p1.poll()    #判断程序进程是否存在，None：表示程序正在运行 其他值：表示程序已退出
                self.poll_2 = self.p2.poll()
                self.poll_3 = self.p3.poll()  
                self.poll_4 = self.p4.poll() 
                self.poll_5 = self.p5.poll() 
                self.poll_6 = self.p6.poll() 
                self.poll_7 = self.p7.poll() 
                self.poll_8 = self.p8.poll() 
                self.poll_9 = self.p9.poll() 
                self.poll_10 = self.p10.poll() 
                if self.poll_1 is None and \
                    self.poll_2 is None and \
                    self.poll_3 is None and \
                    self.poll_4 is None and \
                    self.poll_5 is None and \
                    self.poll_6 is None and \
                    self.poll_7 is None and \
                    self.poll_8 is None and \
                    self.poll_9 is None and \
                    self.poll_10 is None:
                    self.log.logger.info("前{}个程序运行正常".format(self.index))
                else:
                    self.index+=10
                    self.log.logger.info("未检测到程序运行状态，准备启动第{}个程序".format(self.index))
                    if self.index < 100: # 100
                        self.run()
                    else:
                        break
        except KeyboardInterrupt as e:
            self.log.logger.info("检测到CTRL+C，准备退出程序!")

        self.log.logger.info('end.')
 
    def run(self):
        if self.ext == ".py":
            self.log.logger.info('start OK!')
            self.p1 = subprocess.Popen(['python','%s'%self.cmd] + args(self.index), stdin = sys.stdin,stdout = sys.stdout, stderr = sys.stderr, shell = False)
            self.p2 = subprocess.Popen(['python','%s'%self.cmd] + args(self.index+1), stdin = sys.stdin,stdout = sys.stdout, stderr = sys.stderr, shell = False)
            self.p3 = subprocess.Popen(['python','%s'%self.cmd] + args(self.index+2), stdin = sys.stdin,stdout = sys.stdout, stderr = sys.stderr, shell = False)
            self.p4 = subprocess.Popen(['python','%s'%self.cmd] + args(self.index+3), stdin = sys.stdin,stdout = sys.stdout, stderr = sys.stderr, shell = False)
            self.p5 = subprocess.Popen(['python','%s'%self.cmd] + args(self.index+4), stdin = sys.stdin,stdout = sys.stdout, stderr = sys.stderr, shell = False)
            self.p6 = subprocess.Popen(['python','%s'%self.cmd] + args(self.index+5), stdin = sys.stdin,stdout = sys.stdout, stderr = sys.stderr, shell = False)
            self.p7 = subprocess.Popen(['python','%s'%self.cmd] + args(self.index+6), stdin = sys.stdin,stdout = sys.stdout, stderr = sys.stderr, shell = False)
            self.p8 = subprocess.Popen(['python','%s'%self.cmd] + args(self.index+7), stdin = sys.stdin,stdout = sys.stdout, stderr = sys.stderr, shell = False)
            self.p9 = subprocess.Popen(['python','%s'%self.cmd] + args(self.index+8), stdin = sys.stdin,stdout = sys.stdout, stderr = sys.stderr, shell = False)
            self.p10 = subprocess.Popen(['python','%s'%self.cmd] + args(self.index+9), stdin = sys.stdin,stdout = sys.stdout, stderr = sys.stderr, shell = False)
        else:
            pass

if __name__ == '__main__':
    app = Auto_Run(TIME,CMD)