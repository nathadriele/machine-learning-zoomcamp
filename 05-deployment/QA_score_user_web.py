import pickle
import os
from flask import Flask, jsonify, request

app = Flask("score_user")

def get_dv_model():
    dv = load_dv(name="dv.bin")    
    model_name = os.getenv("MODEL_NAME", "model1.bin") 
    print("model_name: ", model_name)
    model = load_model(model_name)
    return dv, modelin)