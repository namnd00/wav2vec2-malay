import malaya_speech
#import numpy as np

SAVED_MODEL_PATH = "wav2vec2-conformer-large"
MAX_LENGTH = 22050
SAMPLE_RATE = 16000

class _Wav2vec2_Predict_Services:

    model = None
    _instance = None

    def predict(self, file_path):
        """
        :param file_path (str): Path to audio file to predict
        :return predicted_keyword (str): Keyword predicted by the model
        """

        # extract MFCC
        signal = self.preprocess(file_path)


        # get the predicted label
        predictions = self.model.predict([signal],decoder = 'beam', beam_size = 5)
        print(predictions)
        predicted_keyword = predictions[0]
        return predicted_keyword


    def preprocess(self, file_path):
        """Extract MFCCs from audio file.
        :param file_path (str): Path of audio file
        """

        # load audio file
        signal, sample_rate = malaya_speech.load(file_path)

        return signal


def Wav2vec2_Predict_Services():
    """Factory function for Keyword_Spotting_Service class.
    :return _Keyword_Spotting_Service._instance (_Keyword_Spotting_Service):
    """

    # ensure an instance is created only the first time the factory function is called
    if _Wav2vec2_Predict_Services._instance is None:
        _Wav2vec2_Predict_Services._instance = _Wav2vec2_Predict_Services()
        _Wav2vec2_Predict_Services.model = malaya_speech.stt.deep_ctc(model = SAVED_MODEL_PATH)
    return _Wav2vec2_Predict_Services._instance


if __name__ == "__main__":

    # create 2 instances of the keyword spotting service
    wps = Wav2vec2_Predict_Services()
    wps1 = Wav2vec2_Predict_Services()

    # check that different instances of the keyword spotting service point back to the same object (singleton)
    assert wps is wps1

    # make a prediction
    word = wps.predict("test1.wav")
    print(word)