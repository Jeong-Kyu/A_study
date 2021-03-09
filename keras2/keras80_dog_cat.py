#data/image/vgg
#개, 고양이 ,라이언, 슈트
#파일명 : dog1.jpg, cat1.jpg, lion1.jpg, suit1.jpg

# C:\data\image\vgg

from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import matplotlib.pyplot as plt

# 이미지 불러오기
img_dog = load_img('../data/image/vgg/dog1.jpg', target_size=(224,224))
img_cat = load_img('../data/image/vgg/cat1.jpg', target_size=(224,224))
img_lion = load_img('../data/image/vgg/lion1.jpg', target_size=(224,224))
img_suit = load_img('../data/image/vgg/suit1.jpg', target_size=(224,224))
# print(img_suit) # <PIL.Image.Image image mode=RGB size=224x224 at 0x1CDF75785B0>
# plt.imshow(img_lion)
# plt.show()

# 이미지 수치화
arr_dog = img_to_array(img_dog)
arr_cat = img_to_array(img_cat)
arr_lion = img_to_array(img_lion)
arr_suit = img_to_array(img_suit)
# print(arr_suit.shape) # (224, 224, 3)
# print(type(arr_suit)) # <class 'numpy.ndarray'>

# RGB -> BGR
from tensorflow.keras.applications.vgg16 import preprocess_input
arr_dog = preprocess_input(arr_dog)
arr_cat = preprocess_input(arr_cat)
arr_lion = preprocess_input(arr_lion)
arr_suit = preprocess_input(arr_suit)
# print(arr_suit.shape) # (224, 224, 3)

arr_input = np.stack([arr_dog, arr_cat, arr_lion, arr_suit])
# print(arr_input.shape) # (4, 224, 224, 3)

# 모델구성
model = VGG16()
results = model.predict(arr_input)

print(results)
print('results.shape : ', results.shape) # results.shape :  (4, 1000) -> 이미지넷에서 구별가능한 카테고리 1000

# 이미지 결과 확인
from tensorflow.keras.applications.vgg16 import decode_predictions
decode_results = decode_predictions(results)
print("=====================================")
print("results[0] : ", decode_results[0])
print("=====================================")
print("results[1] : ", decode_results[1])
print("=====================================")
print("results[2] : ", decode_results[2])
print("=====================================")
print("results[3] : ", decode_results[3])

# =====================================
# results[0] :  [('n02113023', 'Pembroke', 0.78946716), 
#                ('n02113186', 'Cardigan', 0.12041088), 
#                ('n02115641', 'dingo', 0.06430266),   
#                ('n02091467', 'Norwegian_elkhound', 0.010307664),
#                ('n02085620', 'Chihuahua', 0.0037140697)]    
# =====================================
# results[1] :  [('n03887697', 'paper_towel', 0.11095996), 
#                ('n03207941', 'dishwasher', 0.108668),
#                ('n02113023', 'Pembroke', 0.09629205), 
#                ('n04004767', 'printer', 0.061898615), 
#                ('n15075141', 'toilet_tissue', 0.050443273)]      
# =====================================
# results[2] :  [('n04399382', 'teddy', 0.47476104),    
#                ('n03935335', 'piggy_bank', 0.114502005), 
#                ('n03188531', 'diaper', 0.068926856), 
#                ('n03908618', 'pencil_box', 0.04007239), 
#                ('n07930864', 'cup', 0.020324552)]
# =====================================
# results[3] :  [('n04350905', 'suit', 0.5065723), 
#                ('n02883205', 'bow_tie', 0.054467574), 
#                ('n03733281', 'maze', 0.048500672), 
#                ('n03680355', 'Loafer', 0.030960664), 
#                ('n10148035', 'groom', 0.014527284)]
