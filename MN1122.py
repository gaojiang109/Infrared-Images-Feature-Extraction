"""
Created on Mon Oct 15 23:58:37 2018

@author: Zongjiang GAO
"""
#### 少锁定几层,前六层不可训练，看看效果,加上Batch normalizaiton 。序号13. 2019年1月18日10:15:40
###https://medium.com/@14prakash/transfer-learning-using-keras-d804b2e04ef8
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, RMSprop, SGD
from keras import optimizers

from keras.models import Sequential, Model, load_model
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D, BatchNormalization, Reshape,Conv2D, Activation
from keras import backend as K 
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping, History

import numpy as np
from keras.applications import MobileNet
from keras.utils import multi_gpu_model,plot_model
import matplotlib.pyplot as plt
import json

img_width, img_height = 224, 224
train_data_dir = "train"
test_data_dir = "test"
batch_size =100
epochs =200
layer_frozen=2

#设定随机种子
#np.random.seed(1)


### train and test generators with data Augumentation 
datagen = ImageDataGenerator(vertical_flip = True,horizontal_flip=True, fill_mode = "nearest",zoom_range = 0.2,
                                  width_shift_range = 0.2,height_shift_range=0.2,rotation_range=20,validation_split=0.1)

train_generator = datagen.flow_from_directory(train_data_dir,
                                                    target_size = (img_height, img_width),
                                                    batch_size = batch_size, 
                                                    class_mode = "categorical",
                                                    subset='training')
validation_generator = datagen.flow_from_directory(train_data_dir,
                                                    target_size = (img_height, img_width),
                                                    batch_size = batch_size, 
                                                    class_mode = "categorical",
                                                    subset='validation')

test_generator = datagen.flow_from_directory(test_data_dir,
                                                    target_size = (img_height, img_width),
                                                    batch_size = batch_size, 
                                                    class_mode = "categorical")




test_label=np.zeros([1,2])
test_data=np.zeros([1,img_width, img_height ,3])

for j in range(5):
    test_x,test_y=test_generator.next()
    test_data=np.concatenate((test_data,test_x),axis=0)
    test_label=np.concatenate((test_label,test_y))  

test_data=test_data[1:,:,:,:]
test_label=test_label[1:,:] # The first row is zero, deleted

print('size of train is:',len(train_generator))
print('size of validation is:',len(validation_generator))
print('size of test is:',len(test_data))
'''
model1 =MobileNet(include_top=True, weights='imagenet', input_tensor=None, input_shape=(img_width, img_height, 3))
print('inclued top')
model1.summary()
plot_model(model1, to_file='include_top.png')

print('**********************************************************************************************************************************************************************************')
'''
model2 =MobileNet(include_top=False, weights='imagenet', input_tensor=None, input_shape=(img_width, img_height, 3))

x=model2.output

x=GlobalAveragePooling2D()(x)
x=Reshape((1,1,1024))(x)
x=Dropout(0.3)(x)
x=Conv2D(2,1,padding='same')(x)
x=Activation('softmax')(x)
x=Reshape((2,))(x)

model4 = Model(inputs= model2.input, outputs = x)

print('transfer top')
#plot_model(model3, to_file='transfer_top.png')

for layer in model4.layers[:layer_frozen]: #frozen the first layers.
   layer.trainable = False
   
#Adding custom Layers 
#model4 = multi_gpu_model(model3, gpus=5)

print('the last frozen layer is',model4.layers[layer_frozen])

model4.summary()
#optimizer = optimizers.SGD(lr=0.001, decay=1e-6)
#optimizer = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
#sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True) # vgg19_15.h5 val_acc=0.8819
'''
sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True) # 

model4.compile(loss = "categorical_crossentropy", optimizer = sgd, metrics=["accuracy"]) 

# Save the model according to the conditions  
checkpoint = ModelCheckpoint("mobilenet1121.h5", monitor='val_acc', verbose=1,save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto')
history=History()

# display the layer in model
#for i, v in enumerate(model4.layers):
#    print(i,v)
    

history=model4.fit_generator(train_generator,
                          steps_per_epoch=len(train_generator),
                          epochs = epochs,
                          validation_data = validation_generator,
                          validation_steps=len(validation_generator),
                         callbacks = [checkpoint, early, history])



#model4 = load_model('vgg19_15.h5')
score, acc =model4.evaluate(test_data, test_label, batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)


plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['training', 'validation'], loc='upper left')
plt.savefig('mobilenet1121_acc.jpg')
plt.gcf().clear()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('mobilenet1121_his.jpg')

####save history
with open('mobilenet1121_his.json', 'w') as f:
    json.dump(history.history, f)
'''