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
        # H 小时、
        # D 天、
        # W 每星期（interval==0时代表星期一）
        # midnight 每天凌晨
        th.setFormatter(format_str)#设置文件里写入的格式
        self.logger.addHandler(sh) #把对象加到logger里
        self.logger.addHandler(th)

TIME = 1                        #程序状态检测间隔（单位：分钟）
CMD = "/home/jiashuncheng/code/Trasient/ODPA/new/model_torch.py"                 #需要执行程序的绝对路径，支持jar 如：D:\\calc.exe 或者D:\\test.jar

list_ = [0, 20, 40, 60, 80, 100, 120, 140, 160, 180] # [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180]

def args(i):
    _args = ["--exp", "v28_log_0.2_seed4", "--delay", "6", "--gpu", "1", "--num_iterations", "1000", "--seed", "4", "--mode", "test", "--transient", "True", "--connection_prob", "0.2", "--regions", "True", "--hidden_num", "600", "--noise_h", "False", "--lr", "0.001", "--save_model", "--in_", "True", "--out", "True", "--v_m", "4", "--v_gamma", "1", "--scale", "1", "--batch_size", "64", "--alpha", "0.6", "--model_path", "/home/jiashuncheng/code/Trasient/ODPA/new/savedir/exp_v10_tansient_True_connection_0.2_region_True_mode_train_delay_6_hidd_600_seed_4_lr0.001_vm4.0_vgamma1.0_alpha0.6/20240209003227_model_params_step1000.pth", "--sample", "0", "--match", "0", "--theta", "{}".format(list_[i])]
    return _args
 
class Auto_Run():
    def __init__(self,sleep_time,cmd):
        self.log = Logger('log/ts.log',level='debug')
        self.sleep_time = sleep_time
        self.cmd = cmd
        self.ext = (cmd[-3:]).lower()        #判断文件的后缀名，全部换成小写
        self.p = None             #self.p为subprocess.Popen()的返回值，初始化为None

        self.index = 0
        self.args = args(self.index)

        self.run()                           #启动时先执行一次程序
 
        try:
            while 1:
                time.sleep(sleep_time * 5)  #休息1分钟，判断程序状态
                self.poll = self.p.poll()    #判断程序进程是否存在，None：表示程序正在运行 其他值：表示程序已退出
                if self.poll is None:
                    self.log.logger.info("第{}个程序运行正常".format(self.index))
                else:
                    self.index+=1
                    self.log.logger.info("未检测到程序运行状态，准备启动第{}个程序".format(self.index))
                    if self.index < len(list_):
                        self.run()
                    else:
                        break
        except KeyboardInterrupt as e:
            self.log.logger.info("检测到CTRL+C，准备退出程序!")

        self.log.logger.info('end.')
 
    def run(self):
        if self.ext == ".py":
            self.log.logger.info('start OK!')
            self.p = subprocess.Popen(['python','%s'%self.cmd] + args(self.index), stdin = sys.stdin,stdout = sys.stdout, stderr = sys.stderr, shell = False)
        else:
            pass

app = Auto_Run(TIME,CMD) 
