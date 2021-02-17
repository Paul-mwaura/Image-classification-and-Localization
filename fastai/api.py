import aiohttp
from flask import Flask, request
from fastai import *
import requests
from fastai.vision import *
from io import BytesIO

export_file_url = 'https://drive.google.com/uc?export=download&id=1E0_1x9E6EgwHqbOjPx-4rtOKK_pAfowO'
export_file_name = 'models/export.pkl'

classes = [1, 0]

path = Path(__file__).parent

app = Flask(__name__)


def download_file(url, dest):
    if dest.exists():
        return

    response = requests.get(export_file_url)
    with open("models/export.pkl", "wb") as f:
        f.write(response.content)


def setup_learner():
    download_file(export_file_url, path/export_file_name)
    try:
        learn = load_learner(path, export_file_name)
        return learn
    except RuntimeError as e:
        if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
            print(e)
            message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment.\n\nPlease update the fastai library in your training environment and export your model again.\n\nSee instructions for 'Returning to work' at https://course.fast.ai."
            raise RuntimeError(message)


learn = setup_learner()
