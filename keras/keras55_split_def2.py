import numpy as np

a = np.array(range(1,11))
size = 7

def split_x(seq, size):
    aaa = []
    for i in range(len(seq) - size + 1):
        subset = seq[i:(i+size)]
        aaa.append(subset)    #[item for item in subset]
    return np.array(aaa)

dataset = split_x(a,size)
print("=========================")
print(dataset)
dataset = np.transpose(dataset)
print(dataset)

# split 다입력(M:M)
def split_xy5(dataset, x_column, x_low, y_column, y_low):
    x, y = list(), list()
    for i in range(len(dataset)):
        x_end_number = i + x_column
        print(x_end_number, '!!')
        y_end_number = x_end_number + y_column
        print(y_end_number)
        
        if y_end_number > len(dataset):
            break
        tmp_x = dataset[i:x_end_number, i:(i+x_low+1)]
        tmp_y = dataset[x_end_number:y_end_number, (i+x_low+1):i:(i+x_low+2+y_low)]
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)

x, y = split_xy5(dataset, 3, 1,2,2)
print(x, '\n',y)
print("=========================")

dataset = np.array([[1,2,3,4,5,6,7,8,9,10],[11,12,13,14,15,16,17,18,19,20],[21,22,23,24,25,26,27,28,29,30]])
dataset = np.transpose(dataset)

# split 다입력(M:M)
def split_xy4(dataset, x_low, x_col, y_low, y_col):
    x, y = list(), list()
    for i in range(len(dataset)):
        x_end_number = i + x_low
        y_end_number = x_end_number + y_low -1
        if y_end_number > len(dataset):
            break
        tmp_x = dataset[i:x_end_number,:x_col]
        tmp_y = dataset[x_end_number:y_end_number+1,x_col:]
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)

x, y = split_xy4(dataset, 3,2,4,1)
print(x, '\n', y)
print("=========================")