# # 실습
# # outliers1을 행렬 형태에 적용

# import numpy as np

# aaa = np.array([[1,2,3,4,10000,6,7,5000,90,100],[100,200,3,400,500,600,700,8,900,1000]])
# aaa = aaa.transpose()
# print(aaa.shape) #(10, 2)

# def outliers(data_out):
#     quartile_1, q2, quartile_3 = np.percentile(data_out, [25,50,75])
#     print("1사분위 : ", quartile_1)
#     print("q2 : ", q2)
#     print("3사분위 : ", quartile_3)
#     iqr = quartile_3 - quartile_1
#     lower_bound = quartile_1 - (iqr * 1.5)
#     upper_bound = quartile_3 + (iqr * 1.5)
#     print(lower_bound, upper_bound)
#     return np.where((data_out > upper_bound) | (data_out < lower_bound))
    
# outlier_loc = outliers(aaa[:,0])
# outlier_loc1 = outliers(aaa[:,1])

# print("이상치의 위치 : ",outlier_loc) 
# print("이상치의 위치 : ",outlier_loc1) 

# # 1사분위 :  3.25
# # q2 :  6.5
# # 3사분위 :  97.5
# # 이상치의 위치 :  (array([8, 9], dtype=int64),)

# # percentile  알아보기

# import matplotlib.pyplot as plt
# plt.boxplot([aaa[:,0],aaa[:,1]],labels=['0','1'])
# plt.show()


# 실습
# outliers1을 행렬형태도 적용할 수 있도록 수정

import numpy as np

aaa = np.array([[1,2,3,4,10000,6,7,5000,90,100],
                [1000,2000,3,4000,5000,6000,7000,8,9000,10000]])

aaa = aaa.transpose()   
print(aaa.shape)    #(10, 2)

# print(aaa[:,0])

outlier_loc_list = []
def outliers (data_out):
    
    for col in range(data_out.shape[1]):
        print(data_out[:,col])
        quartile_1, quartile_2, quartile_3 = np.percentile(data_out[:,col], [25,50,75])    #percentile:지정된 축을 따라 데이터의 q 번째 백분위 수를 계산합니다.
        print('1사분위 : ', quartile_1)
        print('2사분위 : ', quartile_2)
        print('3사분위 : ', quartile_3)

        iqr = quartile_3 - quartile_1   # 3사분위 - 1사분위
        # 양방향으로 1.5배씩 늘려서 정상적인 데이터범위 지정
        lower_bound = quartile_1 - (iqr * 1.5)  
        upper_bound = quartile_3 + (iqr * 1.5)

        outlier_loc = np.where((data_out[:,col]>upper_bound) | (data_out[:,col]<lower_bound))
        print(outlier_loc)
        outlier_loc_list.append(outlier_loc)
    return np.array(outlier_loc_list)


outlier_loc_list = outliers(aaa)
print('이상치의 위치 : ', outlier_loc_list)

import matplotlib.pyplot as plt
plt.boxplot(aaa)
plt.show() 