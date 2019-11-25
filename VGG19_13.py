"""
Created on Mon Oct 15 23:58:37 2018

@author: Zongjiang GAO
"""
#### 少锁定几层,前六层不可训练，看看效果,加上Batch normalizaiton 。序号13. 2019年1月18日10:15:40
###https://medium.com/@14prakash/transfer-learning-using-keras-d804b2e04ef8
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model, load_model
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D, BatchNormalization
from keras import backend as K 
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping, History

import numpy as np
from keras.applications.vgg19 import preprocess_input
from keras.utils import multi_gpu_model

import matplotlib.pyplot as plt
import json

img_width, img_height = 224, 224
train_data_dir = "train"
test_data_dir = "test"
batch_size =200
epochs =100
layer_frozen=6

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



'''
### to save augumented examples
train_label=np.zeros([1,2])
train_data=np.zeros([1,img_width, img_height ,3])

or i in range(10):
    train_x,train_y=train_generator.next()
    train_data=np.concatenate((train_data,train_x),axis=0)
    train_label=np.concatenate((train_label,train_y))

train_label=train_label[1:,:] # The first row is zero, deleted
train_data=train_data[1:,:,:,:]	

np.save('train_img',train_data)
np.save('train_label',train_label) 
'''

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

print('size of test is:',test_data.shape)


model = applications.VGG19(weights = "imagenet", include_top=False, input_shape = (img_width, img_height, 3))

#Adding custom Layers 

model3 = Sequential()
model3.add(Flatten(input_shape=model.output_shape[1:]))
model3.add(BatchNormalization())
model3.add(Dense(512, activation='relu'))
model3.add(Dropout(0.7))
model3.add(Dense(256, activation="relu"))
model3.add(Dense(2, activation="softmax"))

# creating the final model 
model4 = Model(inputs= model.input, outputs = model3(model.output))

#model4 = multi_gpu_model(model3_1, gpus=5)


for layer in model4.layers[:layer_frozen]: #frozen the first layers.
   layer.trainable = False
   
model4.summary()
#optimizer = optimizers.SGD(lr=0.001, decay=1e-6)
#optimizer = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
#sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True) # vgg19_15.h5 val_acc=0.8819
sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True) # 
model4.compile(loss = "categorical_crossentropy", optimizer = sgd, metrics=["accuracy"]) 

# Save the model according to the conditions  
checkpoint = ModelCheckpoint("vgg19_20.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, 
                             mode='auto', period=1)
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
plt.savefig('accuracy18.jpg')
plt.gcf().clear()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('history18.jpg')

####save history
with open('history18.json', 'w') as f:
    json.dump(history.history, f)



'''

model.save('my_model13.h5')  # creates a HDF5 file 'my_model.h5'
del model  # deletes the existing model
'''
'''
###save the model to json,only the architecture
json_string = model.to_json()
with open("model4.json", "w") as json_file:
    json_file.write(json_string)
'''

#from keras.utils import plot_model
#plot_model(model, to_file='vgg_model.png')

#parallel_model = multi_gpu_model(model2, gpus=5) ###Parallel_model can be used in trainning and testing. But cannot get the right layers.

## compile the model 
#parallel_model.compile(loss = "categorical_crossentropy", optimizer = optimizers.SGD(lr=0.0001, momentum=0.8), metrics=["accuracy"]) #validation accuracy=0.9
#parallel_model.compile(loss = "categorical_crossentropy", 
#                       optimizer = optimizers.RMSprop(lr=0.001, rho=0.7),
#                       metrics=["accuracy"])

'''
#intermediate layer  output, OK 
x=plt.imread('apple1.jpg') 
x2= np.expand_dims(x, axis=0) #add  an axis to the image, because the input should have four aixes.
print(x2.shape)
# with a Sequential model
get_3rd_layer_output = K.function([model3.layers[0].input],[model3.layers[5].output])
layer_output = get_3rd_layer_output([x2])[0]

out=np.sum(np.abs(layer_output[:,:,:,:]),axis=3)
print(out.shape)
print(out)

'''
