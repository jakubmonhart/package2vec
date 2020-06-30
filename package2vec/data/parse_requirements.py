import requests
from lxml import html
import json
import numpy as np
from collections import defaultdict
import pandas as pd
import time
from tqdm import tqdm


response = requests.get("https://pypi.org/simple/")
tree = html.fromstring(response.content)
packages = [package for package in tree.xpath('//a/text()')]

start = time.time()
datadict = defaultdict(list)
for i in tqdm(range(len(packages))):
# for i in tqdm(range(1000)):
    response = requests.get(
        ('https://pypi.org/pypi/{}/json'.format(packages[i])))
    if response.status_code == 200:
        json_package = json.loads(response.content)
        if json_package['info']['requires_dist'] != None:
            for dependency in json_package['info']['requires_dist']:
                datadict['index'].append(i)
                datadict['package'].append(packages[i])
                datadict['dependency'].append(dependency)
        else:
            datadict['index'].append(i)
            datadict['package'].append(packages[i])
            datadict['dependency'].append(np.nan)


print("\njob finished")
end = time.time()
print('elapsed time: {}'.format(end - start))
df = pd.DataFrame(data=datadict)
df = df.dropna()
df.drop(['index'], axis = 1, inplace = True)


df.to_csv('requirements_raw.csv', index=False)
