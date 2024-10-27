import pickle
import os
from flask import Flask, jsonify, request

app = Flask("score_user")