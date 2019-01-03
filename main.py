import numpy as np
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint

import os

def train():
    imagegen = image.ImageDataGenerator()
    input = imagegen.flow_from_directory("inputset",batch_size=16,target_size=(256,256))

    # create the base pre-trained model
    base_model = InceptionResNetV2(weights=None, include_top=False)
    
    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    # and a logistic layer -- let's say we have 200 classes
    predictions = Dense(153, activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    tmp_check = ModelCheckpoint("weights.hdf5", monitor='val_loss', verbose=0, save_best_only=False,
                                save_weights_only=False, mode='auto', period=1)


    if os.path.isfile("weights.hdf5"):
        model.load_weights('weights.hdf5', by_name=True)
        print("load weights..... ok")
    else:
        print("load weights..... no")
        # compile the model (should be done *after* setting layers to non-trainable)
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

        # train the model on the new data for a few epochs
        model.fit_generator(input, steps_per_epoch=None, epochs=10, verbose=1, validation_data=None,
                            validation_steps=None, class_weight=None, max_queue_size=10, workers=1,
                            use_multiprocessing=False,
                            shuffle=True, initial_epoch=0)


    for i, layer in enumerate(base_model.layers):
        print(i, layer.name)

    for layer in model.layers[:249]:
        layer.trainable = False
    for layer in model.layers[249:]:
        layer.trainable = True

    from keras.optimizers import SGD
    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')

    model.fit_generator(input, epochs=100, callbacks=[tmp_check])


def test():
    imagegen = image.ImageDataGenerator()
    input = np.array(imagegen.flow_from_directory("testset", class_mode=None, target_size=(256, 256))[0])
    print(input.shape)

    # create the base pre-trained model
    base_model = InceptionResNetV2(weights=None, include_top=False)

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    # and a logistic layer -- let's say we have 200 classes
    predictions = Dense(153, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    model.load_weights('weights.hdf5', by_name=True)

    output = model.predict(input,verbose=0, steps=None)
    output = list(output)
    with open("labels_pred.txt", 'w') as f:
        for out in output:
            out = list(out)
            acc = max(out)
            if acc >= 0.75:
                f.write("{}\n".format(out.index(acc)))
            else:
                f.write("Unclassified\n")

if __name__ == "__main__":
    is_train = False
    if is_train:
        train()
    else:
        test()