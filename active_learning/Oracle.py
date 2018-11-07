import numpy as np
import subprocess
from tqdm import tqdm

def __get_point(arr):
    assert arr.shape[0] == 10
    point = ' '.join([str(i) for i in arr.tolist()])
    process = './data/Oracle.static ' + point
    result = subprocess.check_output(process, shell=True)
    result = float(result)
    return result

def get_points(arr):
    pt_size = arr.shape[0]
    ans = []
    for pt_id in tqdm(range(pt_size)):
        pt = arr[pt_id, :]
        ans.append(__get_point(pt))
    return np.array(ans)