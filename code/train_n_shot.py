import tkinter as tk
import threading
import numpy as np
import wave
import librosa
import matplotlib.pyplot as plt
import os
import pyaudio
from pathlib import Path
from typing import Tuple
from tkinter import messagebox
from multilingual_kws.embedding import transfer_learning, input_data
import sounddevice as sd
import tensorflow as tf

class AudioProcessor:
    @staticmethod
    def select_highest_energy_segment(audio: np.array, sample_rate: int, window_size: float = 1.0, overlap: float = 0.95) -> Tuple[int, int]:
        window_size_samples = int(window_size * sample_rate)
        overlap_samples = int(overlap * sample_rate)
        step_size = window_size_samples - overlap_samples

        max_energy = 0
        max_start_index = 0

        for i in range(0, len(audio) - window_size_samples + 1, step_size):
            segment = audio[i:i + window_size_samples]
            energy = np.sum(np.square(segment))

            if energy > max_energy:
                max_energy = energy
                max_start_index = i

        start = max_start_index
        end = start + window_size_samples

        return start, end

    @staticmethod
    def plot_segment(start: int, end: int, audio: np.array, sample_rate: int):
        duration = len(audio) / sample_rate
        time = np.linspace(0, duration, len(audio))

        plt.figure(figsize=(10, 4))
        plt.plot(time, audio, label='Audio Signal', color='blue')
        plt.axvspan(start / sample_rate, end / sample_rate, color='red', alpha=0.3, label='Selected Segment')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.title('Audio Signal with Segment of Highest Energy')
        plt.legend()
        plt.show()

    @staticmethod
    def cut_and_pad_wav(input_file, output_file, start_sample, stop_sample):
        with wave.open(input_file, 'rb') as wf:
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            framerate = wf.getframerate()

            start_byte = start_sample * n_channels * sampwidth
            stop_byte = stop_sample * n_channels * sampwidth

            wf.setpos(start_sample)
            frames = wf.readframes(stop_sample - start_sample)

        samples_per_second = framerate * n_channels
        frame_count = len(frames) // (n_channels * sampwidth)

        if frame_count < samples_per_second:
            zero_samples_needed = samples_per_second - frame_count
            zero_frames = b'\x00' * zero_samples_needed * n_channels * sampwidth
            frames += zero_frames

        Path(output_file).parent.mkdir(exist_ok=True, parents=True)

        with wave.open(output_file, 'wb') as wf_out:
            wf_out.setnchannels(n_channels)
            wf_out.setsampwidth(sampwidth)
            wf_out.setframerate(framerate)
            wf_out.writeframes(frames)

class KWSModelTrainer:
    @staticmethod
    def finetune_kws(model_path: str, data_dir: str, background_noise: str, unknown_files_txt: str) -> str:
        unknown_files = []

        with open(Path(data_dir) / "saved_text.txt", "r") as file:
            keyword = str(file.readline())

        with open(unknown_files_txt, "r") as fh:
            unknown_files = [str(Path(unknown_files_txt).parent / w) for w in fh.read().splitlines()]

        print("Number of unknown files:", len(unknown_files))

        n_samples = [str(file) for file in (Path(data_dir) / "wavs").glob("*.wav")]
        print(n_samples)
        train_files = n_samples[1:]
        dev_samples = [n_samples[0]]

        print("---Training model---")
        model_settings = input_data.standard_microspeech_model_settings(3)
        _, model, _ = transfer_learning.transfer_learn(
            target=keyword,
            train_files=train_files,
            val_files=dev_samples,
            unknown_files=unknown_files,
            num_epochs=4,
            num_batches=1,
            batch_size=64,
            primary_lr=0.001,
            backprop_into_embedding=False,
            embedding_lr=0,
            model_settings=model_settings,
            base_model_path="./content/embedding_model/multilingual_context_73_0.8011",
            base_model_output="dense_2",
            UNKNOWN_PERCENTAGE=50.0,
            bg_datadir=background_noise,
            csvlog_dest=None,
        )
        model_path = f"./data_n_shot/{keyword}_{len(train_files)}_shot"
        model.save(model_path)
        print(f"Model saved as {model_path}")
        return model_path

class AudioRecorderApp:
    def __init__(self, root: tk.Tk, input_device_index: int, model_path: str, background_noise: str, unknown_files_txt: str):
        self.root = root
        self.input_device_index = input_device_index
        self.root.title("Audio Recorder with Prediction")
        self.num_wav = 0
        self.is_recording = False
        self.audio_chunks = []
        self.record_thread = None

        Path("data_n_shot").mkdir(parents=True, exist_ok=True)

        self.setup_ui()

    def setup_ui(self):
        self.entry = tk.Entry(self.root, width=50)
        self.entry.pack(pady=10)
        self.save_button = tk.Button(self.root, text="Save Text", command=self.save_text)
        self.save_button.pack(pady=5)

        self.record_button = tk.Button(self.root, text="Start Recording", command=self.toggle_recording)
        self.record_button.pack(pady=20)

        self.remove_button = tk.Button(self.root, text="Remove last file", command=self.remove_file)
        self.remove_button.pack(pady=30)

        self.finetune_button = tk.Button(self.root, text="Finetune model", command=self.finetune_model)
        self.finetune_button.pack(pady=40)

        self.exit_button = tk.Button(self.root, text="Exit", command=self.root.destroy)
        self.exit_button.pack(pady=50)

    def save_text(self):
        text = self.entry.get()
        if text:
            file_path = os.path.join("data_n_shot", "saved_text.txt")
            with open(file_path, 'w') as file:
                file.write(text)
            messagebox.showinfo("Success", f"Text saved successfully to {file_path}")
        else:
            messagebox.showwarning("Warning", "No text to save")

    def toggle_recording(self):
        if self.is_recording:
            self.is_recording = False
            self.record_button.config(text="Start Recording")
            self.record_thread.join()
            self.process_recorded_audio()
        else:
            self.is_recording = True
            self.record_button.config(text="Stop Recording")
            self.audio_chunks = []
            self.record_thread = threading.Thread(target=self.record_audio)
            self.record_thread.start()
            self.num_wav += 1

    def remove_file(self):
        if self.num_wav > 0:
            file_path = f"./data_n_shot/wavs/wav_{self.num_wav}.wav"
            if os.path.exists(file_path):
                os.remove(file_path)
                self.num_wav -= 1

    def record_audio(self):
        CHUNK = 1600
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 16000

        p = pyaudio.PyAudio()
        stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK, input_device_index=self.input_device_index)

        frames = []

        try:
            while self.is_recording:
                data = stream.read(CHUNK)
                frames.append(data)
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()

            audio_data = b''.join(frames)
            Path("data_n_shot/wavs").mkdir(exist_ok=True, parents=True)
            with wave.open(f"./data_n_shot/wavs/wav_{self.num_wav}.wav", 'wb') as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(p.get_sample_size(FORMAT))
                wf.setframerate(RATE)
                wf.writeframes(audio_data)

    def process_recorded_audio(self):
        current_filename = f"./data_n_shot/wavs/wav_{self.num_wav}.wav"
        audio, sample_rate = librosa.load(current_filename, sr=None)
        start, end = AudioProcessor.select_highest_energy_segment(audio, sample_rate)
        AudioProcessor.plot_segment(start, end, audio, sample_rate)
        AudioProcessor.cut_and_pad_wav(current_filename, current_filename, start, end)

    def finetune_model(self):
        KWSModelTrainer.finetune_kws(
            model_path=model_path,
            data_dir="data_n_shot",
            background_noise=background_noise,
            unknown_files_txt=unknown_files_txt,
        )

if __name__ == "__main__":
    print(sd.query_devices())
    input_device_index = -1

    model_path = "./content/embedding_model/multilingual_context_73_0.8011"
    background_noise = "../speech_commands/_background_noise_/"
    unknown_files_txt = "../unknown_files.txt"

    root = tk.Tk()
    app = AudioRecorderApp(root, input_device_index, model_path, background_noise, unknown_files_txt)
    root.mainloop()
