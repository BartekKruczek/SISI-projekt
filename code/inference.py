import pyaudio
import numpy as np
from multilingual_kws.embedding import input_data
import tensorflow as tf

class RealTimePrediction:
    def __init__(self, model_path, chunk=1600, format=pyaudio.paInt16, channels=1, rate=16000):
        self.chunk = chunk
        self.format = format
        self.channels = channels
        self.rate = rate
        self.frames = []
        self.model = tf.keras.models.load_model(model_path)
        self.settings = input_data.standard_microspeech_model_settings(label_count=1)
        self.init_audio_stream()

    def init_audio_stream(self):
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=self.format,
                                  channels=self.channels,
                                  rate=self.rate,
                                  input=True,
                                  frames_per_buffer=self.chunk)

    def make_prediction(self, audio: np.array):
        spectrogram = np.expand_dims(input_data.to_micro_spectrogram(self.settings, audio), axis=0)
        predictions = self.model.predict(spectrogram, verbose=None)
        categorical_predictions = np.argmax(predictions, axis=1)
        return predictions, categorical_predictions

    def record_and_predict(self):
        print("* recording")
        try:
            while True:
                data = self.stream.read(self.chunk)
                data = np.frombuffer(data, dtype=np.int16) / np.iinfo(np.int16).max
                self.frames.append(data)
                
                if len(self.frames) > 10:
                    self.frames.pop(0)

                if self.frames:
                    infer_second = np.concatenate(self.frames)
                    if infer_second.shape[0] < 16000:
                        infer_second = np.pad(infer_second, (0, 16000 - infer_second.shape[0]))
                    
                    predictions, categorical_predictions = self.make_prediction(infer_second)
                    print(predictions)
        finally:
            self.cleanup()

    def cleanup(self):
        print("* done recording")
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()

if __name__ == "__main__":
    model_path = "modelpath"
    RealTimePrediction(model_path).record_and_predict()
