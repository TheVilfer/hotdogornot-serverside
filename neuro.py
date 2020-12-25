import json
import os

import numpy as np
import tensorflow as tf
from PIL import Image

MODEL_DIR = os.path.dirname(__file__)


class Model(object):
    def __init__(self, model_dir=MODEL_DIR):

        model_path = os.path.realpath(model_dir)
        if not os.path.exists(model_path):
            raise ValueError(
                f"Exported model folder doesn't exist {model_dir}")
        self.model_path = model_path

        with open(os.path.join(model_path, "signature.json"), "r") as f:
            self.signature = json.load(f)
        self.inputs = self.signature.get("inputs")
        self.outputs = self.signature.get("outputs")

        self.session = None

    def load(self):
        self.cleanup()

        self.session = tf.compat.v1.Session(graph=tf.Graph())

        tf.compat.v1.saved_model.loader.load(
            sess=self.session, tags=self.signature.get("tags"), export_dir=self.model_path)

    def predict(self, image: Image.Image):

        if self.session is None:
            self.load()

        width, height = image.size

        if width != height:
            square_size = min(width, height)
            left = (width - square_size) / 2
            top = (height - square_size) / 2
            right = (width + square_size) / 2
            bottom = (height + square_size) / 2

            image = image.crop((left, top, right, bottom))

        if "Image" not in self.inputs:
            raise ValueError(
                "Couldn't find Image in model inputs - please report issue to Lobe!")
        input_width, input_height = self.inputs["Image"]["shape"][1:3]
        if image.width != input_width or image.height != input_height:
            image = image.resize((input_width, input_height))

        image = np.asarray(image) / 255.0

        feed_dict = {self.inputs["Image"]["name"]: [image]}

        fetches = [(key, output["name"])
                   for key, output in self.outputs.items()]

        outputs = self.session.run(
            fetches=[name for _, name in fetches], feed_dict=feed_dict)

        results = {}

        for i, (key, _) in enumerate(fetches):
            val = outputs[i].tolist()[0]
            if isinstance(val, bytes):
                val = val.decode()
            results[key] = val
        return results

    def cleanup(self):

        if self.session is not None:
            self.session.close()
            self.session = None

    def __del__(self):
        self.cleanup()


def tn(image):

    if image.mode != "RGB":
        image = image.convert("RGB")
    model = Model()
    model.load()
    outputs = model.predict(image)
    return outputs
