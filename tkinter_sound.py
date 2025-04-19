import tkinter as tk
import threading
import sounddevice as sd
from scipy.io.wavfile import write
import whisper
import requests
import time

# ========== 基本設定 ==========
RECORD_SECONDS = 5
AUDIO_FILENAME = "recorded.wav"
WHISPER_MODEL = "medium"  # 可換成 large、medium
LANGCHAIN_API_URL = "http://localhost:8000/ask/invoke"  # 請依實際設定修改


# ========== 功能函式 ==========
def record_audio(filename=AUDIO_FILENAME, duration=RECORD_SECONDS, fs=16000, update_timer=None):
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    for i in range(duration, 0, -1):
        if update_timer:
            update_timer(f"⏱️ 錄音中：{i} 秒")
        time.sleep(1)
    sd.wait()
    write(filename, fs, recording)
    if update_timer:
        update_timer("✅ 錄音完成")

def transcribe_audio(filename):
    model = whisper.load_model(WHISPER_MODEL)
    result = model.transcribe(filename)
    return result['text']

def call_langchain_api(question_text):
    try:
        response = requests.post(LANGCHAIN_API_URL, json={
            "input": {
                "question": question_text,
                "template_name": "professional"
            }
        })
        result = response.json()
        return result.get("output", "❌ 回傳結果異常")
    except Exception as e:
        return f"❌ 呼叫失敗：{e}"

# ========== 主邏輯 ==========
def start_process():
    def task():
        # Step 1: 錄音中
        record_button.config(state=tk.DISABLED, bg="red", text="錄音中...")
        result_label.config(text="")
        transcription_label.config(text="")
        timer_label.config(text="")

        record_audio(update_timer=lambda txt: timer_label.config(text=txt))

        # Step 2: 語音辨識
        timer_label.config(text="🧠 正在轉文字...")
        try:
            transcribed_text = transcribe_audio(AUDIO_FILENAME)
            transcription_label.config(text=f"📝 語音文字：{transcribed_text}")
        except Exception as e:
            transcription_label.config(text=f"❌ 語音辨識失敗：{e}")
            record_button.config(state=tk.NORMAL, bg="SystemButtonFace", text="開始錄音")
            return

        # Step 3: 呼叫 AI
        timer_label.config(text="🤖 AI 回應中...")
        try:
            answer = call_langchain_api(transcribed_text)
            result_label.config(text=f"📣 AI 回覆：\n{answer}")
        except Exception as e:
            result_label.config(text=f"❌ API 錯誤：{e}")

        # Step 4: 結束
        timer_label.config(text="✅ 完成")
        record_button.config(state=tk.NORMAL, bg="SystemButtonFace", text="開始錄音")

    threading.Thread(target=task).start()

# ========== UI 設定 ==========
window = tk.Tk()
window.title("🎤 語音問答小幫手")
window.geometry("500x400")

record_button = tk.Button(window, text="開始錄音", command=start_process, height=2, width=20)
record_button.pack(pady=10)

timer_label = tk.Label(window, text="等待中...", fg="blue")
timer_label.pack(pady=5)

transcription_label = tk.Label(window, text="", wraplength=460, justify="left", fg="darkgreen")
transcription_label.pack(pady=10)

result_label = tk.Label(window, text="", wraplength=460, justify="left", fg="black")
result_label.pack(pady=10)

window.mainloop()