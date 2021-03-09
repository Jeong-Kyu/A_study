# 이상치 처리
# 1. 0 처리
# 2. Nan처리 후 보간
# 3.4.5... 등등

import numpy as np

aaa = np.array([1,2,3,4,6,9,10,50,100])

def outliers(data_out):
    quartile_1, q2, quartile_3 = np.percentile(data_out, [25,50,75])
    print("1사분위 : ", quartile_1)
    print("q2 : ", q2)
    print("3사분위 : ", quartile_3)
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    return np.where((data_out > upper_bound) | (data_out < lower_bound))

outlier_loc = outliers(aaa)
print("이상치의 위치 : ",outlier_loc) 

# 1사분위 :  3.25
# q2 :  6.5
# 3사분위 :  97.5
# 이상치의 위치 :  (array([8, 9], dtype=int64),)

# percentile  알아보기

import matplotlib.pyplot as plt
plt.boxplot(aaa)
plt.show()