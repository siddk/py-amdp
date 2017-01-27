import numpy as np
import pickle
import time
import urllib
import urllib2

times = []
with open("../cross_val/raw_data.pik", 'r') as f:
    data, _ = pickle.load(f)

for k in data:
    for en, _ in data[k]:
        command = urllib.urlencode({"command": " ".join(en)})
        start_time = time.time()
        ans = urllib2.urlopen('http://127.0.0.1:5000/model?' + command).read()
        times.append(time.time() - start_time)

times = np.array(times)
print 'Mean:', np.mean(times)
print 'Stddev:', np.std(times)