import pyaudio
import numpy as np
import requests
import time
import wave
import os
import streamlit as st
import matplotlib.pyplot as plt
import queue
from flask import Flask, request, jsonify
from datetime import datetime
from threading import Thread

# Parameter für die Audioaufnahme
CHUNK = 1024  # Anzahl der Samples pro Frame
FORMAT = pyaudio.paInt16  # 16-bit Audioformat
CHANNELS = 1  # Mono
RATE = 44100  # Abtastrate in Hz
DEFAULT_THRESHOLD = 500  # Standard-Lautstärkeschwelle
API_URL = "https://example.com/api/signal"  # Ziel-URL für das Signal
AUDIO_SAVE_PATH = "\\netzwerkpfad\\audio"  # Speicherort für Audiodateien
LOCAL_SAVE_PATH = "./audio"  # Lokaler Speicherort für Audiodateien

timestamp_queue = queue.Queue(maxsize=50)
volume_queue = queue.Queue(maxsize=50)
threshold = DEFAULT_THRESHOLD

# Liste zur Speicherung der erkannten Geräusche mit Timestamp
detected_noises = []

app = Flask(__name__)


def save_audio(data, filename):
    """Speichert die Audioaufnahme als WAV-Datei auf dem Netzwerk oder lokal."""
    try:
        save_path = AUDIO_SAVE_PATH if os.path.exists(AUDIO_SAVE_PATH) else LOCAL_SAVE_PATH
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        filepath = os.path.join(save_path, filename)
        wf = wave.open(filepath, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(pyaudio.PyAudio().get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(data)
        wf.close()
        print(f"Audio gespeichert unter: {filepath}")
    except Exception as e:
        print(f"Fehler beim Speichern der Audiodatei: {e}")


def send_signal():
    """Sendet ein Signal an die API."""
    data = {"message": "Lautes Geräusch erkannt"}
    try:
        response = requests.post(API_URL, json=data)
        if response.status_code == 200:
            print("Signal erfolgreich gesendet.")
        else:
            print(f"Fehler beim Senden des Signals: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"Fehler: {e}")


def reset_threshold():
    """Setzt die Lautstärkeschwelle zurück."""
    global threshold
    threshold = DEFAULT_THRESHOLD
    print("Lautstärkeschwelle zurückgesetzt auf Standardwert.")


def calibrate_threshold():
    """Misst die Standardlautstärke für 10 Sekunden."""
    global threshold
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

    st.info("Kalibrierung läuft... Bitte leise sein.")
    progress_bar = st.progress(0)
    values = []
    start_time = time.time()
    while time.time() - start_time < 10:
        data = np.frombuffer(stream.read(CHUNK, exception_on_overflow=False), dtype=np.int16)
        volume = np.linalg.norm(data)
        values.append(volume)
        progress_bar.progress(int((time.time() - start_time) / 10 * 100))

    threshold = int(np.mean(values) * 1.8)
    st.success(f"Neue Lautstärkeschwelle: {threshold}")

    stream.stop_stream()
    stream.close()
    p.terminate()


@app.route('/set_threshold', methods=['POST'])
def set_threshold():
    """API-Endpunkt zum Setzen der Lautstärkeschwelle."""
    global threshold
    data = request.json
    threshold = data.get("threshold", DEFAULT_THRESHOLD)
    return jsonify({"message": "Threshold gesetzt", "threshold": threshold})


@app.route('/reset_threshold', methods=['POST'])
def api_reset_threshold():
    """API-Endpunkt zum Zurücksetzen der Lautstärkeschwelle."""
    reset_threshold()
    return jsonify({"message": "Threshold zurückgesetzt", "threshold": threshold})


@app.route('/get_threshold', methods=['GET'])
def get_threshold():
    """API-Endpunkt zum Abrufen der aktuellen Lautstärkeschwelle."""
    return jsonify({"threshold": threshold})


@app.route('/get_detected_noises', methods=['GET'])
def get_detected_noises():
    """API-Endpunkt zum Abrufen der erkannten Geräusche."""
    return jsonify({"detected_noises": detected_noises})


def main():
    """Hauptfunktion zur Geräuscherkennung."""
    st.title("Geräuscherkennungssystem")
    global threshold

    # Sidebar für Einstellungen
    with st.sidebar:
        st.header("Einstellungen")
        threshold = st.slider("Empfindlichkeit der Erkennung", 100, 50000, threshold)

        if st.button("Kalibrieren (10 Sek)"):
            calibrate_threshold()

        if st.button("Lautstärkeschwelle zurücksetzen"):
            reset_threshold()
            st.success("Lautstärkeschwelle zurückgesetzt.")

    # Hauptbereich
    st.header("Echtzeit-Lautstärkeverlauf")
    chart_placeholder = st.empty()

    # Buffer für die letzten 10 Sekunden
    audio_buffer = []
    buffer_size = int(RATE / CHUNK * 10)  # 10 Sekunden

    # Start/Stop-Schaltfläche
    if "recording" not in st.session_state:
        st.session_state.recording = False

    if st.button("Aufnahme starten" if not st.session_state.recording else "Aufnahme stoppen"):
        st.session_state.recording = not st.session_state.recording

    if st.session_state.recording:
        p = pyaudio.PyAudio()
        stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

        try:
            while st.session_state.recording:
                data = stream.read(CHUNK, exception_on_overflow=False)
                np_data = np.frombuffer(data, dtype=np.int16)
                volume = np.linalg.norm(np_data)

                if timestamp_queue.full():
                    timestamp_queue.get()
                    volume_queue.get()

                timestamp_queue.put(time.time())
                volume_queue.put(volume)

                # Füge die aktuellen Daten zum Buffer hinzu
                audio_buffer.append(data)
                if len(audio_buffer) > buffer_size:
                    audio_buffer.pop(0)

                # Echtzeit-Lautstärkeverlauf
                fig, ax = plt.subplots()
                ax.plot(list(timestamp_queue.queue), list(volume_queue.queue), color='blue', label="Lautstärke")
                ax.axhline(y=threshold, color='red', linestyle='--', label="Schwelle")
                ax.set_title("Lautstärkeverlauf")
                ax.set_xlabel("Zeit")
                ax.set_ylabel("Lautstärke")
                ax.legend()
                chart_placeholder.pyplot(fig)

                if volume > threshold:
                    st.warning("Lautes Geräusch erkannt!")
                    send_signal()

                    # Speichere den Timestamp des erkannten Geräuschs
                    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    detected_noises.append(timestamp)

                    # Kombiniere die letzten 10 Sekunden (before) und die nächsten 10 Sekunden (after)
                    combined_data = b''.join(audio_buffer)  # Before-Daten
                    audio_buffer.clear()

                    # Warte 10 Sekunden und sammle die nächsten 10 Sekunden (after)
                    for _ in range(buffer_size):
                        data = stream.read(CHUNK, exception_on_overflow=False)
                        combined_data += data  # After-Daten hinzufügen

                    # Speichere die kombinierten Daten in einer Wave-Datei
                    filename = f"geraeusch_{timestamp}.wav"
                    save_thread = Thread(target=save_audio, args=(combined_data, filename))
                    save_thread.start()

                    # Zeige die letzte Überschreitung an
                    st.metric("Letzte Überschreitung", timestamp.replace('_', ' '))

                    # Warte, bis die Datei vollständig geschrieben wurde
                    save_thread.join()  # Warte auf das Ende des Speicherthreads

                    # Vorschau der gespeicherten Audiodatei
                    if os.path.exists(os.path.join(LOCAL_SAVE_PATH, filename)):
                        st.audio(os.path.join(LOCAL_SAVE_PATH, filename), format="audio/wav")

                # Verzögerung, um den Graph flüssiger zu halten
                time.sleep(0.05)  # Reduzierte Verzögerung

        except KeyboardInterrupt:
            st.stop()
            print("Beende Geräuscherkennung...")
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()


if __name__ == "__main__":
    from threading import Thread

    Thread(target=lambda: app.run(host='0.0.0.0', port=5000, debug=False)).start()
    main()