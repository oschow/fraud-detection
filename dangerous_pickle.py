import cPickle as pickle
from models import logistic as models
# import whatever model we want

# model = model_name

def pickle_model(model):
    with open("model.pkl",'w') as f:
        pickle.dump(model, f)

def unpickle_model(model):
    with open("model.pkl") as f_un:
        model_unpickled = pickle.load(f_un)
