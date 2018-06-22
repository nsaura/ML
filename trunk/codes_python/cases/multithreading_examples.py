#from thread import start_new_thread

#thread.start_new_thread ( function, args[, kwargs] )
#http://www.techbeamers.com/python-multithreading-concepts/

import threading
import datetime

class myThread (threading.Thread):
    def __init__(self, name, counter):
        threading.Thread.__init__(self)
        self.threadID = counter
        self.name = name
        self.counter = counter
    def run(self, function, *args):
        print ("Starting " + self.name)
        print function(*args)
        print ("Exiting " + self.name)

def print_date(threadName, counter):
    datefields = []
    today = datetime.date.today()
    datefields.append(today)
    print ("%s[%d]: %s" % ( threadName, counter, datefields[0] ))

# Create new threads
thread1 = myThread("Thread", 1)
thread2 = myThread("Thread", 2)

# Start new Threads
# Starting method has been overrided by run
#thread1.run(lambda x : 5*x, 5)
#thread2.run(lambda x : 5*x, 65)

thread1.start(lambda x : 5*x, 5)
thread2.start(lambda x : 5*x, 5)

thread1.join()
thread2.join()
print ("Exiting the Program!!!")
