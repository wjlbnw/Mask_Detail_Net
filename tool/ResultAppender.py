import os
import time

class ResultAppender:
    def __init__(self, file_name, session_name, mod='a'):
        self.file_name = file_name
        self.session_name = session_name
        self.mod = mod
        self.write_line('\n'+self.session_name+'\t'+str(time.ctime()))

    def write(self, text):
        _file = open(self.file_name, mode=self.mod)
        _file.write(text)
        _file.close()
    def write_line(self, text):
        self.write(text+'\n')