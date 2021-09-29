import requests
import time
# server url
URL = "http://127.0.0.1:5000/predict"


# audio file we'd like to send for predicting keyword
FILE_PATH = "D:/speed_to_text/code/wav2vec2-malay/tests/waves/fp6ABAjcXPI.0050.wav"


if __name__ == "__main__":
    start = time.time()
    # open files
    file = open(FILE_PATH, "rb")

    # package stuff to send and perform POST request
    values = {"file": (FILE_PATH, file, "audio/wav")}
    response = requests.post(URL, files=values)
    data = response.json()
    stop = time.time()
    print("Predicted word: {}".format(data["word"]))
    print("pred Time: %0.2f s" %(stop-start))