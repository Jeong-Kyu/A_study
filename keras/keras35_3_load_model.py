import numpy as np

a = np.array(range(1,11))
size = 5

def split_x(seq, size):
    aaa = []
    for i in range(len(seq) - size + 1):
        subset = seq[i:(i+size)]
        aaa.append(subset)    #[item for item in subset]
    return np.array(aaa)

dataset = split_x(a,size)

print(dataset)

x = dataset[:,:4]
y = dataset[:,-1]

print(x.shape)
print(y.shape)

x = x.reshape(6,4,1)


from tensorflow.keras.models import load_model
model = load_model("../data/h5/save_keras35.h5")


model.compile(loss = 'mse', optimizer= 'adam', metrics=['mae'])
model.fit(x, y, epochs=1000, batch_size=10)


loss = model.evaluate(x,y, batch_size=1)

print("model.metrics_name : ", model.metrics_names)
print(loss)

x_pred = np.array([7,8,9,10])
x_pred = x_pred.reshape(1,4,1)
y_pred = model.predict(x_pred)
print(y_pred)

# model.metrics_name :  ['loss', 'mae']
# [2.897868398576975e-05, 0.0046115717850625515]
# [[10.944815]]