import sys
import numpy as np
from tensorflow import keras
import os

#pop off the classify.py
sys.argv.pop(0)
model = keras.models.load_model(sys.argv.pop(0), compile=False)
spot = 0
for val in sys.argv:
    #predict the outcome with the model
    if (os.path.exists(sys.argv[spot])):
        path = sys.argv[spot]
        img = keras.preprocessing.image.load_img(path, target_size=(100, 100, 3))
        arrImage = keras.utils.img_to_array(img)
        npArr = np.array([arrImage])

        predictions = model.predict(npArr)
        spot += 1
        if predictions[0][0] > 0.5:
            print(f'{path} is a dog')
        else:
            print(f'{path} is a cat')
    else:
        print(f'Couldnt Find a File Named {sys.argv[spot]}')
        spot += 1


    #print the prediction result