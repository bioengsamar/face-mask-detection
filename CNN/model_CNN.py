from keras.models import model_from_json
import numpy as np

class face_mask_detection_Model(object):
    
    classes_LIST = ["with mask", "without mask"]
    def __init__(self, model_json_file, model_weights_file):
        # load model from JSON file
        with open(model_json_file, "r") as json_file:
            loaded_model_json = json_file.read()
            self.loaded_model = model_from_json(loaded_model_json)

        # load weights into the new model
        self.loaded_model.load_weights(model_weights_file)
        #self.loaded_model._make_predict_function()
        
    def predict_class(self, img):
        self.preds = self.loaded_model.predict(img)
        print(self.preds)
        return self.preds
        #if self.preds >= 0.2:
            
            #return face_mask_detection_Model.classes_LIST[1]
        #else:
            #return face_mask_detection_Model.classes_LIST[0]