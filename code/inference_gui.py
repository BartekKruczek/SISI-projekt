import pyaudio
import numpy as np
import matplotlib.pyplot as plt
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
        self.init_plot()

    def init_audio_stream(self):
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=self.format,
                                  channels=self.channels,
                                  rate=self.rate,
                                  input=True,
                                  frames_per_buffer=self.chunk)

    def init_plot(self):
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.x = np.arange(0, 100)
        self.y = np.zeros(100)
        self.line, = self.ax.plot(self.x, self.y)
        self.ax.set_ylim(0, 1)
        self.ax.set_xlim(0, 100)
        self.ax.set_xlabel('Time (frames)')
        self.ax.set_ylabel('Prediction')
        self.ax.set_title('Real-time Prediction')

    def make_prediction(self, audio: np.array):
        spectrogram = np.expand_dims(input_data.to_micro_spectrogram(self.settings, audio), axis=0)
        predictions = self.model.predict(spectrogram, verbose=None)
        categorical_predictions = np.argmax(predictions, axis=1)
        return predictions, categorical_predictions

    def update_plot(self, prediction):
        self.y = np.roll(self.y, -1)
        self.y[-1] = prediction
        self.line.set_ydata(self.y)
        self.ax.draw_artist(self.ax.patch)
        self.ax.draw_artist(self.line)
        self.fig.canvas.flush_events()
        plt.pause(0.01)

    def run(self):
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

                    predictions, _ = self.make_prediction(infer_second)
                    target_prediction = predictions[0][2]
                    print(f"silence/background noise {predictions[0][0]}, unknown keyword:  {predictions[0][1]}, target:  {target_prediction}")

                    self.update_plot(target_prediction)
        finally:
            print("* done recording")
            self.stream.stop_stream()
            self.stream.close()
            self.p.terminate()
            plt.ioff()
            plt.show()

if __name__ == "__main__":
    model_path = "modelpath"
    RealTimePrediction(model_path).run()
