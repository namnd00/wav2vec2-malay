import random
import os
from flask import Flask, request, jsonify
from predicts.wav2vec2_predict_services import Wav2vec2_Predict_Services


# instantiate flask app
app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def predict():
	"""Endpoint to predict keyword
    :return (json): This endpoint returns a json file with the following format:
        {
            "word": "satu dua tiga"
        }
	"""

	# get file from POST request and save it
	audio_file = request.files["file"]
	file_name = str(random.randint(0, 100000))
	audio_file.save(file_name)

	# instantiate keyword spotting service singleton and get prediction
	wps = Wav2vec2_Predict_Services()
	predicted_word = wps.predict(file_name)
	print(predicted_word)
	# we don't need the audio file any more - let's delete it!
	os.remove(file_name)

	# send back result as a json file
	result = {"word": predicted_word}
	#result = {"word": "hello"}

	return jsonify(result)


if __name__ == "__main__":
    app.run(debug=False)