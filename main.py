import numpy as np
import matplotlib.pyplot as plt
import os
import json
from tqdm import tqdm
from bimvee.exportIitYarp import encodeEvents24Bit
data_dir = '/mnt/Shared/'

path = os.path.join(data_dir, 'tuna')


if __name__ == '__main__':

    numBins = 10
    step = 1 / (numBins)
    dirList = os.listdir(path)
    prev_ts = None
    events = []
    ev_img_list = [x for x in dirList if x.__contains__('ec') and os.path.splitext(x)[-1] == '.png']

    for file in tqdm(sorted(ev_img_list)):
        with open(os.path.join(path, file.split('.')[0] + '.json'), 'r') as jsonFile:
            metadata = json.load(jsonFile)
        timestamp = metadata['timestamp']
        if prev_ts is None:
            prev_ts = timestamp
            continue
        image = plt.imread(os.path.join(path, file))
        vCount = (np.round(image / step) - numBins / 2).astype(int)
        vIndices = vCount.nonzero()
        for y, x in zip(vIndices[0], vIndices[1]):
            num_events = vCount[y, x]
            for v in range(abs(num_events)):
                polarity = 1 if num_events > 0 else 0
                ts_noise = (np.random.rand() - 0.5) / 250
                ts = prev_ts + v * ((timestamp - prev_ts) / abs(num_events)) + ts_noise
                events.append([x, y, ts, polarity])
        prev_ts = timestamp
    events = np.array(events)
    events = events[events[:, 2].argsort()]
    prev_ts = events[0, 2]

    # dataDict = {'x': events[:, 0], 'y': events[:, 1], 'ts': events[:, 2], 'pol': events[:, 3].astype(bool)}
    encodedData = np.array(encodeEvents24Bit(events[:, 2] - events[0, 2], events[:, 0], events[:, 1], events[:, 3].astype(bool))).astype(np.uint32)
    with open(os.path.join(os.getenv('HOME'), 'Desktop', 'test', 'binaryevents.log'), 'wb') as f:
        encodedData.tofile(f)
