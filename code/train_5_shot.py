# # import numpy as np
# # import matplotlib.pyplot as plt
# # import librosa
# # import librosa.display

# # def plot_segment(filename, window_size=1.0, overlap=0.5):
# #     # Load audio file
# #     audio, sample_rate = librosa.load(filename, sr=None, mono=True)

# #     # Select segment with highest energy
# #     start, end = select_highest_energy_segment(audio, sample_rate, window_size, overlap)

# #     # Calculate time axis
# #     duration = len(audio) / sample_rate
# #     time = np.linspace(0, duration, len(audio))

# #     # Plot the entire audio waveform
# #     plt.figure(figsize=(10, 4))
# #     plt.plot(time, audio, label='Audio Signal', color='blue')

# #     # Highlight the selected segment
# #     plt.axvspan(start/sample_rate, end/sample_rate, color='red', alpha=0.3, label='Selected Segment')

# #     # Add labels and legend
# #     plt.xlabel('Time (s)')
# #     plt.ylabel('Amplitude')
# #     plt.title('Audio Signal with Segment of Highest Energy')
# #     plt.legend()

# #     # Show the plot
# #     plt.show()

# #     # Return the start and end indices of the segment with highest energy
# #     return start, end

# # idx = np.random.randint(len(df_dataset))
# # plot_segment(df_dataset["wavpath"].iloc[idx], window_size=1.0, overlap=0.5)
# # print(df_dataset["category"].iloc[idx])


# from multilingual_kws.embedding import transfer_learning, input_data

# import tensorflow as tf
# import numpy as np 
# import IPython
# from pathlib import Path
# import matplotlib.pyplot as plt
# import os
# import subprocess
# import csv
# from tqdm.notebook import tqdm


# import csv
# from pathlib import Path



# KEYWORD="off" 

# LANG="en" # ISOCODE for spanish
# KEYWORD="off"
# background_noise = "../speech_commands/_background_noise_/"
# unknown_files_txt = "../unknown_files.txt"
# unknown_files=[]
# with open(unknown_files_txt, "r") as fh:
#     for w in fh.read().splitlines():
#         unknown_files.append("../" + w)
# print("Number of unknown files", len(unknown_files))

# print("---Training model---")
# model_settings = input_data.standard_microspeech_model_settings(3)
# _, model, _ = transfer_learning.transfer_learn(
#     target=KEYWORD,
#     train_files=five_samples,
#     val_files=dev_samples,
#     unknown_files=unknown_files,
#     num_epochs=4,
#     num_batches=1,
#     batch_size=64,
#     primary_lr=0.001,
#     backprop_into_embedding=False,
#     embedding_lr=0,
#     model_settings=model_settings,
#     base_model_path="./content/embedding_model/multilingual_context_73_0.8011",
#     base_model_output="dense_2",
#     UNKNOWN_PERCENTAGE=50.0,
#     bg_datadir=background_noise,
#     csvlog_dest=None,
# )
# model.save(f"{KEYWORD}_5shot")



import tkinter as tk
from tkinter import filedialog, messagebox
import pyaudio
import numpy as np
import matplotlib.pyplot as plt
# from multilingual_kws.embedding import input_data
# import tensorflow as tf
import threading
import wave
from pathlib import Path

# # Load the pre-trained model
# model_path = "/home/piotrek/Documents/personal/STUDIA_DS/semestr_1/PP/multilingual_kws/off_5shot"
# model = tf.keras.models.load_model(model_path)

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Audio Recorder with Prediction")
        self.num_wav = 0

        # Text entry and button to save text to file
        self.entry = tk.Entry(root, width=50)
        self.entry.pack(pady=10)
        self.save_button = tk.Button(root, text="Save Text", command=self.save_text)
        self.save_button.pack(pady=5)

        # Button to start and stop recording
        self.record_button = tk.Button(root, text="Start Recording", command=self.toggle_recording)
        self.record_button.pack(pady=20)

        self.is_recording = False
        self.record_thread = None

        # Matplotlib figure for real-time plotting
        self.fig, self.ax = plt.subplots()
        self.x = np.arange(0, 100)
        self.y = np.zeros(100)
        self.line, = self.ax.plot(self.x, self.y)

        self.ax.set_ylim(-1, 1)
        self.ax.set_xlim(0, 100)
        self.ax.set_xlabel('Time (frames)')
        self.ax.set_ylabel('Amplitude')
        self.ax.set_title('Recording')
        self.fig.canvas.draw()

    def save_text(self):
        text = self.entry.get()
        file_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt")])
        if file_path:
            with open(file_path, 'w') as file:
                file.write(text)
            messagebox.showinfo("Success", "Text saved successfully")

    def toggle_recording(self):
        if self.is_recording:
            self.is_recording = False
            self.record_button.config(text="Start Recording")
        else:
            self.is_recording = True
            self.record_button.config(text="Stop Recording")
            self.record_thread = threading.Thread(target=self.record_audio)
            self.record_thread.start()
            self.num_wav += 1

    def update_plot(self, data_):
        data_length = len(data_)
        if data_length > 100:
            data_ = data_[:100]
        elif data_length < 100:
            data_ = np.pad(data_, (0, 100 - data_length), 'constant')
        
        self.y = np.roll(self.y, -100)
        self.y[-100:] = data_
        self.line.set_ydata(self.y)
        self.ax.draw_artist(self.ax.patch)
        self.ax.draw_artist(self.line)
        self.fig.canvas.flush_events()

    def record_audio(self):
        CHUNK = 1600
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 16000

        p = pyaudio.PyAudio()
        stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

        frames = []

        try:
            while self.is_recording:
                data = stream.read(CHUNK)
                data_ = np.frombuffer(data, dtype=np.int16) / np.iinfo(np.int16).max
                frames.append(data)

                self.root.after(0, self.update_plot, data_)

        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()

            audio = b''.join(frames)
            if len(audio) < 16000:
                audio += b'\x00' * (16000 - len(audio))

            Path("audio").mkdir(exist_ok=True, parents=True)
            wf = wave.open(f"./audio/wav_{self.num_wav}.wav", 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(audio)
            wf.close()

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
