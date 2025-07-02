import os
import torch
import time
from datetime import datetime
from pathlib import Path
from flask import Flask, request, jsonify, render_template
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import shutil
import threading
from functools import lru_cache

app = Flask(__name__)

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
    },
    "mistral": {
        "name": "mistralai/Mistral-7B-v0.1",
        "description": "Мощная 7B модель",
        "quantized": True
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
        """Создает необходимые директории"""
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

    def load_model(self, model_name):
        if model_name not in MODELS_CONFIG:
            raise ValueError(f"Модель {model_name} не найдена в конфигурации")

        if model_name in self.loaded_models:
            self.current_model = model_name
            return self.loaded_models[model_name]

        print(f"Загрузка модели {model_name}...")
        model_config = MODELS_CONFIG[model_name]
        device = self.get_device()

        try:
            # Конфигурация квантования
            quant_config = None
            if model_config["quantized"] and device == "cuda":
                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16
                )

            # Загрузка модели
            model = AutoModelForCausalLM.from_pretrained(
                model_config["name"],
                quantization_config=quant_config,
                device_map="auto" if device == "cuda" else "cpu",
                torch_dtype=torch.float16 if device == "cuda" else torch.float32
            )

            tokenizer = AutoTokenizer.from_pretrained(model_config["name"])

            with self.lock:
                self.loaded_models[model_name] = {
                    "model": model,
                    "tokenizer": tokenizer,
                    "config": model_config
                }
                self.current_model = model_name

            return self.loaded_models[model_name]

        except Exception as e:
            print(f"Ошибка загрузки модели: {str(e)}")
            raise

    def clear_model_cache(self):
        """Очищает папку model_cache"""
        try:
            cache_dir = self.dirs["model_cache"]
            if cache_dir.exists():
                shutil.rmtree(cache_dir)
                cache_dir.mkdir()
                return True
            return False
        except Exception as e:
            print(f"Ошибка очистки кеша: {str(e)}")
            return False

# Инициализация менеджера моделей
model_manager = ModelManager()

# ==============================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ==============================================

def generate_response(prompt, model_name):
    """Генерация ответа с сохранением отчета"""
    try:
        start_time = time.time()

        # Получаем модель
        model_data = model_manager.load_model(model_name)
        model = model_data["model"]
        tokenizer = model_data["tokenizer"]

        # Генерация ответа
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_length=200,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        elapsed = time.time() - start_time

        # Сохранение отчета
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = model_manager.dirs["reports_dir"] / f"report_{timestamp}.txt"

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"Model: {model_name}\n")
            f.write(f"Prompt: {prompt}\n")
            f.write(f"Time: {elapsed:.2f}s\n\n")
            f.write(answer)

        return {
            "success": True,
            "response": answer,
            "time_sec": elapsed,
            "report_path": str(report_path),
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
    model_name = data.get('model')

    try:
        model_manager.load_model(model_name)
        return jsonify({
            "success": True,
            "current_model": model_name
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400

@app.route('/api/generate', methods=['POST'])
def generate():
    if not request.is_json:
        return jsonify({
            "success": False,
            "error": "Request must be JSON"
        }), 400

    data = request.get_json()
    prompt = data.get('prompt', '').strip()
    model_name = data.get('model', model_manager.current_model)

    if not prompt:
        return jsonify({
            "success": False,
            "error": "Prompt cannot be empty"
        }), 400

    result = generate_response(prompt, model_name)

    if result['success']:
        return jsonify(result)
    else:
        return jsonify(result), 500

@app.route('/api/clear_cache', methods=['POST'])
def clear_cache():
    try:
        success = model_manager.clear_model_cache()
        return jsonify({
            "success": success,
            "message": "Model cache cleared successfully"
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

# ==============================================
# ЗАПУСК СЕРВЕРА
# ==============================================

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)