import sys
import keras
import numpy as np
import pandas as pd
import keras.backend as K
from keras.utils import np_utils
from keras.datasets import cifar100
from keras.models import Sequential
from keras.models import load_model
from keras import regularizers, optimizers
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.initializers import glorot_normal, RandomNormal, Zeros
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from sklearn.feature_extraction.image import extract_patches_2d

def patch_preprocess(image):
	# extract a random crop from the image with the target width
	# and height
	return extract_patches_2d(image, (image.shape[0], image.shape[1]),max_patches=1)[0]

# Data Retrieval & mean/std preprocess
# (x_train, y_train), (x_test, y_test) = cifar100.load_data()
# x_train = x_train.astype('float32')
# x_test = x_test.astype('float32')
def data_reader(path,train=False):
	data = pd.read_csv(path,header=None,sep=" ")
	num_classes=100

	x = np.array(data.iloc[:,:-2])
	x = x.reshape((x.shape[0],3,32,32))
	x = np.rollaxis(x,1,4)
	if(train):
		y = data.iloc[:,-1]
		y = np.array(y)
		y = np_utils.to_categorical(y,num_classes)

	# mean = np.mean(x,axis=(0,1,2,3))
	# std = np.std(x,axis=(0,1,2,3))
	# x = (x-mean)/(std+1e-7)
	x = x/255

	if(train):
		return x,y
	else:
		return x

# data = pd.read_csv("./A3_Data/train.csv",header=None,sep=" ")
# x_train = np.array(data.iloc[:,:-2])
# y_train = (data.iloc[:,-1])
# #y_train = pd.get_dummies(y_train)
# y_train = np.array(y_train)
# x_train = x_train.reshape((x_train.shape[0],3,32,32))
# x_train = np.rollaxis(x_train,1,4)
# #dataset = npy.loadtxt(sys.argv[1])
# #x_train = dataset[:,-1:]
# #y_train = dataset[:,:-1]
# data = pd.read_csv("./A3_Data/test.csv",header=None,sep=" ")
# x_test = np.array(data.iloc[:,:-2])
# x_test = x_test.reshape((x_test.shape[0],3,32,32))
# x_test = np.rollaxis(x_test,1,4)
# #dataset = npy.loadtxt(sys.argv[2])
# #x_test = dataset[:,-1:]


# mean = np.mean(x_train,axis=(0,1,2,3))
# std = np.std(x_train,axis=(0,1,2,3))
# x_train = (x_train-mean)/(std+1e-7)
# #x_test = (x_test-mean)/(std+1e-7)

# mean = np.mean(x_test,axis=(0,1,2,3))
# std = np.std(x_test,axis=(0,1,2,3))
# x_test = (x_test-mean)/(std+1e-7)

# num_classes = 100
# y_train = np_utils.to_categorical(y_train,num_classes)
#y_test = np_utils.to_categorical(y_test,num_classes)


# Data augmentation
# datagen = ImageDataGenerator(
#     rotation_range=15,
#     width_shift_range=0.1,
#     height_shift_range=0.1,
#     horizontal_flip=True
#     )
# datagen.fit(x_train)

# Define Model architecture
def create_model(s = 2, weight_decay = 1e-2, act="relu"):
    model = Sequential()
    num_classes = 100
    # Block 1
    model.add(Conv2D(64, (3,3), padding='same', kernel_initializer=glorot_normal(), input_shape=x_train.shape[1:]))
    model.add(BatchNormalization())
    model.add(Activation(act))
    model.add(Dropout(0.2))
    
    # Block 2
    model.add(Conv2D(128, (3,3), padding='same', kernel_initializer=glorot_normal()))
    model.add(BatchNormalization())
    model.add(Activation(act))
    model.add(Dropout(0.2))
    
    # Block 3
    model.add(Conv2D(128, (3,3), padding='same', kernel_initializer=RandomNormal(stddev=0.01)))
    model.add(BatchNormalization())
    model.add(Activation(act))
    model.add(Dropout(0.2))
    
    # Block 4
    model.add(Conv2D(128, (3,3), padding='same', kernel_initializer=RandomNormal(stddev=0.01)))
    model.add(BatchNormalization())
    model.add(Activation(act))
    # First Maxpooling
    model.add(MaxPooling2D(pool_size=(2,2), strides=s))
    model.add(Dropout(0.2))
    
    
    # Block 5
    model.add(Conv2D(128, (3,3), padding='same', kernel_initializer=RandomNormal(stddev=0.01)))
    model.add(BatchNormalization())
    model.add(Activation(act))
    model.add(Dropout(0.2))
    
    # Block 6
    model.add(Conv2D(128, (3,3), padding='same', kernel_initializer=glorot_normal()))
    model.add(BatchNormalization())
    model.add(Activation(act))
    model.add(Dropout(0.2))
    
    # Block 7
    model.add(Conv2D(256, (3,3), padding='same', kernel_initializer=glorot_normal()))
    # Second Maxpooling
    model.add(MaxPooling2D(pool_size=(2,2), strides=s))
    model.add(BatchNormalization())
    model.add(Activation(act))
    model.add(Dropout(0.2))
    
    
    # Block 8
    model.add(Conv2D(256, (3,3), padding='same', kernel_initializer=glorot_normal()))
    model.add(BatchNormalization())
    model.add(Activation(act))
    model.add(Dropout(0.2))
    
    # Block 9
    model.add(Conv2D(256, (3,3), padding='same', kernel_initializer=glorot_normal()))
    model.add(BatchNormalization())
    model.add(Activation(act))
    model.add(Dropout(0.2))
    # Third Maxpooling
    model.add(MaxPooling2D(pool_size=(2,2), strides=s))
    
    
    # Block 10
    model.add(Conv2D(512, (3,3), padding='same', kernel_initializer=glorot_normal()))
    model.add(BatchNormalization())
    model.add(Activation(act))
    model.add(Dropout(0.1))

    # Block 11  
    model.add(Conv2D(2048, (1,1), padding='same', kernel_initializer=glorot_normal()))
    model.add(Activation(act))
    model.add(Dropout(0.1))
    
    # Block 12  
    model.add(Conv2D(256, (1,1), padding='same', kernel_initializer=glorot_normal()))
    model.add(Activation(act))
    # Fourth Maxpooling
    model.add(MaxPooling2D(pool_size=(2,2), strides=s))
    model.add(Dropout(0.1))


    # Block 13
    model.add(Conv2D(256, (3,3), padding='same', kernel_initializer=glorot_normal()))
    model.add(Activation(act))
    # Fifth Maxpooling
    model.add(MaxPooling2D(pool_size=(2,2), strides=s))
    model.add(Dropout(0.2))
    # Final Classifier
    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))

    return model

if __name__ == "__main__":
	# Prepare for training 
    num_classes=50
    train_path = sys.argv[1]
    test_path = sys.argv[2]
    out_path = sys.argv[3]
    x_train,y_train = data_reader(train_path,train=True)
    x_test = data_reader(test_path)
    datagen = ImageDataGenerator(
    rotation_range=15,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    fill_mode='nearest',
    horizontal_flip=True,
    preprocessing_function=patch_preprocess
    )
    datagen.fit(x_train)    
    model = create_model(act="elu")
    batch_size = 256
    epochs = 35
    train = {}
    # First training for 50 epochs - (0-50)
    opt_adm = keras.optimizers.Adadelta()
    model.compile(loss='categorical_crossentropy', optimizer=opt_adm, metrics=['accuracy'])
    train["part_1"] = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
    									steps_per_epoch=x_train.shape[0] // batch_size,epochs=60,
    									verbose=2)
#   model.save("simplenet_generic_first.h5")
    print(train["part_1"].history)
    # Training for 25 epochs more - (50-75)
    opt_adm = keras.optimizers.Adadelta(lr=0.7, rho=0.9)
    model.compile(loss='categorical_crossentropy', optimizer=opt_adm, metrics=['accuracy'])
    train["part_2"] = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
    									steps_per_epoch=x_train.shape[0] // batch_size,epochs=epochs,
    									verbose=2)
#   model.save("simplenet_generic_second.h5")
    print(train["part_2"].history)  
    # Training for 25 epochs more - (75-100)
    opt_adm = keras.optimizers.Adadelta(lr=0.5, rho=0.85)
    model.compile(loss='categorical_crossentropy', optimizer=opt_adm, metrics=['accuracy'])
    train["part_3"] = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
    									steps_per_epoch=x_train.shape[0] // batch_size,epochs=epochs,
    									verbose=2)
#   model.save("simplenet_generic_third.h5")
    print(train["part_3"].history)
    # Training for 25 epochs more  - (100-125)
    opt_adm = keras.optimizers.Adadelta(lr=0.3, rho=0.75)
    model.compile(loss='categorical_crossentropy', optimizer=opt_adm, metrics=['accuracy'])
    train["part_4"] = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
    									steps_per_epoch=x_train.shape[0] // batch_size,epochs=epochs,
                                       verbose=2)
# 	model.save("simplenet_generic_fourth.h5")
    print(train["part_4"].history)

    opt_adm = keras.optimizers.Adadelta(lr=0.1, rho=0.65)
    model.compile(loss='categorical_crossentropy', optimizer=opt_adm, metrics=['accuracy'])
    train["part_4"] = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
    									steps_per_epoch=x_train.shape[0] // batch_size,epochs=epochs,
    									verbose=2)
# 	model.save("simplenet_generic_fourth.h5")
    print(train["part_4"].history)
    predicts = []
    for i in range(10):
            preds = model.predict_generator(datagen.flow(x_test, batch_size=64, shuffle=False),steps=(len(x_test)/64))
            predicts.append(preds)
    final_pred = np.mean(predicts,axis=0)
    class_pred = np.argmax(final_pred,axis=-1)
    #np.savetxt("final_pred.txt",final_pred,delimiter='\n')
    np.savetxt(out_path,class_pred,delimiter='\n')
    

