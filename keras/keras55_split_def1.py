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

# split 다입력(M:M)
def split_xy4(dataset, x_len, y_len):
    x, y = list(), list()
    for i in range(len(dataset)):
        x_end_number = i + x_len
        y_end_number = x_end_number + y_len -1
        if y_end_number > len(dataset):
            break
        tmp_x = dataset[i:x_end_number,:-1]
        print(tmp_x)
        tmp_y = dataset[x_end_number-1:y_end_number,-1]
        print(tmp_y)
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)

x, y = split_xy4(dataset, 2, 4)
print(x, '\n', y)
print("=========================")
'''
# split 다입력(M:M)
dataset = np.transpose(dataset)

def split_xy(dataset, x_len, y_len):
    x, y = list(), list()
    for i in range(len(dataset)):
        x_end_number = i + x_len
        y_end_number = x_end_number + y_len -1
        if y_end_number > len(dataset):
            break
        tmp_x = dataset[i:x_end_number,:-1]
        tmp_y = dataset[x_end_number-1:y_end_number,-1]
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)

x, y = split_xy(dataset, 5, 2)
print(x, '\n',y)



dataset = np.array([1,2,3,4,5,6,7,8,9,10])
# split (M:1)

def split_xy1(dataset, time_steps) :
    x, y = list(), list()
    for i in range(len(dataset)):
        end_number = i + time_steps
        if end_number > len(dataset) -1:
            break
        tmp_x = dataset[i:end_number]
        tmp_y = dataset[end_number]
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)

x,y = split_xy1(dataset,5)
print(x, '\n',y)

# split (M:M)

def split_xy2(dataset, time_steps, y_column):
    x, y = list(), list()
    for i in range(len(dataset)):
        x_end_number = i + time_steps
        y_end_number = x_end_number + y_column
        if y_end_number > len(dataset):
            break
        tmp_x = dataset[i:x_end_number]
        tmp_y = dataset[x_end_number : y_end_number]
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)

x,y = split_xy2(dataset,5,2)
print(x, '\n',y)
'''
# split 다입력(M:1)
dataset = np.array([[1,2,3,4,5,6,7,8,9,10],[11,12,13,14,15,16,17,18,19,20],[21,22,23,24,25,26,27,28,29,30]])
dataset = np.transpose(dataset)
print(len(dataset))
'''
def split_xy3(dataset, time_steps, y_column):
    x, y = list(), list()
    for i in range(len(dataset)):
        x_end_number = i + time_steps
        y_end_number = x_end_number + y_column -1
        if y_end_number > len(dataset):
            break
        tmp_x = dataset[i:x_end_number,:-1]
        tmp_y = dataset[x_end_number-1:y_end_number,-1]
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)

x, y = split_xy3(dataset, 4, 1)
print(x, '\n',y)
print("=========================")
'''
# split 다입력(M:M)
def split_xy4(dataset, time_steps, y_column):
    x, y = list(), list()
    for i in range(len(dataset)):
        x_end_number = i + time_steps
        y_end_number = x_end_number + y_column -1
        if y_end_number > len(dataset):
            break
        tmp_x = dataset[i:x_end_number,:-1]
        tmp_y = dataset[x_end_number-1:y_end_number,-1]
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)

x, y = split_xy4(dataset, 2, 4)
print(x, '\n', y)
print("=========================")

# split 다입력(M:M)
def split_xy5(dataset, time_steps, y_column):
    x, y = list(), list()
    for i in range(len(dataset)):
        x_end_number = i + time_steps
        y_end_number = x_end_number + y_column
        if y_end_number > len(dataset):
            break
        tmp_x = dataset[i:x_end_number, :]
        tmp_y = dataset[x_end_number:y_end_number, :]
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)

x, y = split_xy5(dataset, 2, 2)
print(x, '\n',y)
print("=========================")
