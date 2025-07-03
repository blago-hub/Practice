import os
import torch
import time
from datetime import datetime
from pathlib import Path
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import shutil
import threading
from functools import lru_cache

app = Flask(__name__)
CORS(app)  # Включение CORS

# ==============================================
# КОНФИГУРАЦИЯ МОДЕЛЕЙ
# ==============================================

MODELS_CONFIG = {
    "tinyllama": {
        "name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "description": "Компактная модель (1.1B параметров)",
        "quantized": True
    },
    "phi2": {
        "name": "microsoft/phi-2",
        "description": "Модель от Microsoft (2.7B параметров)",
        "quantized": False
    }
}

# ==============================================
# КЛАСС ДЛЯ УПРАВЛЕНИЯ МОДЕЛЯМИ
# ==============================================

class ModelManager:
    def __init__(self):
        self.loaded_models = {}
        self.current_model = None
        self.lock = threading.Lock()
        self.dirs = self.setup_directories()

    def setup_directories(self):
        base_dir = Path.cwd()
        model_cache = base_dir / "model_cache"
        reports_dir = base_dir / "reports"

        model_cache.mkdir(parents=True, exist_ok=True)
        reports_dir.mkdir(parents=True, exist_ok=True)

        return {
            "model_cache": model_cache,
            "reports_dir": reports_dir
        }

    def get_device(self):
        return "cuda" if torch.cuda.is_available() else "cpu"

    @lru_cache(maxsize=3)
    def load_tokenizer(self, model_name):
        return AutoTokenizer.from_pretrained(MODELS_CONFIG[model_name]["name"])

    def load_model(self, model_name):
        if model_name not in MODELS_CONFIG:
            raise ValueError(f"Модель {model_name} не найдена")

        if model_name in self.loaded_models:
            self.current_model = model_name
            return self.loaded_models[model_name]

        print(f"Загрузка модели {model_name}...")
        model_config = MODELS_CONFIG[model_name]
        device = self.get_device()

        try:
            quant_config = None
            if model_config["quantized"] and device == "cuda":
                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16
                )

            model = AutoModelForCausalLM.from_pretrained(
                model_config["name"],
                quantization_config=quant_config,
                device_map="auto" if device == "cuda" else None,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32
            )

            tokenizer = self.load_tokenizer(model_name)

            with self.lock:
                self.loaded_models[model_name] = {
                    "model": model,
                    "tokenizer": tokenizer,
                    "config": model_config
                }
                self.current_model = model_name

            return self.loaded_models[model_name]

        except Exception as e:
            print(f"Ошибка загрузки: {str(e)}")
            raise

    def clear_model_cache(self):
        try:
            shutil.rmtree(self.dirs["model_cache"])
            self.dirs["model_cache"].mkdir()
            return True
        except Exception as e:
            print(f"Ошибка очистки кеша: {str(e)}")
            return False

model_manager = ModelManager()

# ==============================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ==============================================

def generate_response(prompt, model_name):
    try:
        if len(prompt) > 2000:
            raise ValueError("Слишком длинный промт (максимум 2000 символов)")

        start_time = time.time()
        model_data = model_manager.load_model(model_name)
        
        inputs = model_data["tokenizer"](prompt, return_tensors="pt").to(model_data["model"].device)
        outputs = model_data["model"].generate(
            **inputs,
            max_length=150,
            temperature=0.7,
            top_p=0.9
        )
        
        answer = model_data["tokenizer"].decode(outputs[0], skip_special_tokens=True)
        elapsed = time.time() - start_time

        # Сохранение отчета
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = model_manager.dirs["reports_dir"] / f"report_{timestamp}.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"Model: {model_name}\nPrompt: {prompt}\nTime: {elapsed:.2f}s\n\n{answer}")

        return {
            "success": True,
            "response": answer,
            "time_sec": elapsed,
            "model": model_name
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

# ==============================================
# API ЭНДПОИНТЫ
# ==============================================

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/health')
def health_check():
    return jsonify({"status": "ok"}), 200

@app.route('/api/models', methods=['GET'])
def get_models():
    return jsonify({
        "available_models": list(MODELS_CONFIG.keys()),
        "current_model": model_manager.current_model,
        "device": model_manager.get_device()
    })

@app.route('/api/switch_model', methods=['POST'])
def switch_model():
    data = request.get_json()
    if not data or 'model' not in data:
        return jsonify({"success": False, "error": "Неверный запрос"}), 400
        
    try:
        model_manager.load_model(data['model'])
        return jsonify({"success": True, "current_model": data['model']})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

@app.route('/api/generate', methods=['POST'])
def generate():
    if not request.is_json:
        return jsonify({"success": False, "error": "Требуется JSON"}), 400

    data = request.get_json()
    prompt = data.get('prompt', '').strip()
    model_name = data.get('model', model_manager.current_model)

    if not prompt:
        return jsonify({"success": False, "error": "Промт не может быть пустым"}), 400

    result = generate_response(prompt, model_name)
    status_code = 200 if result['success'] else 500
    return jsonify(result), status_code

@app.route('/api/clear_cache', methods=['POST'])
def clear_cache():
    try:
        success = model_manager.clear_model_cache()
        return jsonify({"success": success, "message": "Кеш очищен"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

# ==============================================
# ЗАПУСК СЕРВЕРА
# ==============================================

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
