from tqdm import tqdm
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
import numpy as np
import cv2


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

# https://keras.io/applications/
def featurize_images(filesnames, model, batch_size=32):
    features = []
    
    for chunk in tqdm(chunks(filesnames, batch_size)):
        load_img = []
        for fname in chunk:
            img = image.load_img(fname, target_size=(224, 224))
            x = image.img_to_array(img)

            # img = cv2.imread(fname)
            # img = cv2.resize(img,(224, 224)).astype(np.float32)

            x = np.expand_dims(x, axis=0)
            load_img.append(preprocess_input(x))
        preds = model.predict_on_batch(np.concatenate(load_img)).squeeze()
        # make preds multidimensional even for 1 sized batch
        if len(preds.shape)==1:
            preds = np.expand_dims(preds, axis=0)
        features.extend(preds)
    return np.array(features)
