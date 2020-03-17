# 必要的一些常用函数
import csv
import datetime
import os

# 不需要写成类的工具函数


class util():
    # csv读取
    def csvRead(csvPath):
        csvFile = csv.reader(open(csvPath, 'r', encoding="utf8"))
        return csvFile

    # csv读取（详细版）
    def csvReadDetail(csvPath):
        try:
            csvFile = csv.reader(open(csvPath, 'r'))
        except Exception as e:
            print("Error: %s,\t,%s" % (Exception, e))
            print("There may be some problem in %s" % csvPath)
            return ""
        dataList = []
        for csvLine in csvFile:
            dataList.append(csvLine)
        return dataList

    # csv写入
    def csvWrite(csvPath):
        csvFile = open(csvPath, 'w', encoding='utf8', newline='')
        writer = csv.writer(csvFile)
        return writer

    def csvInit():
        csv.field_size_limit(1000000000)
        return 0

    def datetime_toString(dt):
        return dt.strftime("%Y-%m-%d-%H")

    def mkdir(path):

        # 判断路径是否存在
        # 存在     True
        # 不存在   False
        isExists = os.path.exists(path)

        # 判断结果
        if not isExists:
            # 如果不存在则创建目录
            # 创建目录操作函数
            os.makedirs(path)

            print
            path + ' 创建成功'
            return True
        else:
            # 如果目录存在则不创建，并提示目录已存在
            print
            path + ' 目录已存在'
            return False

# 时间工具


class timingTool():
    startTime = 0

    def __init__(self):
        self.startTime = datetime.datetime.now()

    def timmingGet(self):
        return datetime.datetime.now() - self.startTime

    def getNow(self):
        return datetime.datetime.now()

# 记录工具
class logTool():

    path = ""
    reader = ""

    def __init__(self, path):
        print(path)
        self.path = path
        self.open()
        self.close()

    def writer(self, note):
        self.open()
        print(note, file=self.reader)
        self.close()

    def open(self):
        self.reader = open(self.path, "a")

    def close(self):
        self.reader.close()

    def info(self, note):
        self.open()
        print("[INFO:%s] : %s" %
              (datetime.datetime.now(), note), file=self.reader)
        self.close()

    def error(self, note):
        self.open()
        print("[ERROR:%s] : %s" %
              (datetime.datetime.now(), note), file=self.reader)
        self.close()
