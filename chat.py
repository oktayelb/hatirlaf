import json
from llama_cpp import Llama

# 1. Load the Model (Loads once into VRAM)
model_path = "./Qwen2.5-7B-Instruct-Q4_K_M.gguf"

print("Loading model into VRAM...")
llm = Llama(
    model_path=model_path,
    n_ctx=2048,
    n_gpu_layers=-1, 
    verbose=False
)

# 2. Target Paragraph
paragraph = """
Bugün 21 Nisan 2026, Ve ben şu anda yürüyüşteyim. Geçen hafta bugün emre alp , ozan ve ben mezunlar derneğinin etkinliğine gittik. Orada neva da vardı, hatta Bugün onunla buluşacağım.
"""

# ==========================================
# PASS 1: The Logical Extraction (Text-to-Text)
# ==========================================
print("\nPass 1: Analyzing logic and temporal data...")

pass_1_system_prompt = """Sen Contextual Voice Diary uygulaması için çalışan bir veri analiz motorusun.
Kullanıcının metnini oku ve tespit ettiğin her farklı olayı liste halinde dök.
Her olay için Tarih, Zaman Dilimi (Geçmiş/Şu An/Gelecek), Lokasyon, Olay ve Kişiler bilgilerini çıkar.
Kritik Kurallar:
- Metinde belirtilmeyen mekanlar veya kişiler için kesinlikle "Bilinmeyen Lokasyon" veya "Bilinmeyen Kişi" yaz.
- Söyleyen kişiyi "Ben" olarak ekle.
- "Geçen hafta" gibi ifadelerin tam tarihini o günün tarihine göre hesapla."""

pass_1_messages = [
    {"role": "system", "content": pass_1_system_prompt},
    {"role": "user", "content": paragraph}
]

pass_1_response = llm.create_chat_completion(
    messages=pass_1_messages,
    temperature=0.1,
    # No JSON constraint here. Let the model "think" freely.
)

structured_text = pass_1_response["choices"][0]["message"]["content"]
print("\n--- Pass 1 Output (Raw Logic) ---")
print(structured_text)


# ==========================================
# PASS 2: The Formatting (Text-to-JSON)
# ==========================================
print("\nPass 2: Formatting into strict JSON...")

# We no longer need the "dusunce_sureci" key because Pass 1 handled the thinking.
json_schema = {
    "type": "object",
    "properties": {
        "olay_loglari": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "zaman_dilimi": {
                        "type": "string",
                        "enum": ["Geçmiş", "Şu An", "Gelecek"]
                    },
                    "tarih": {"type": "string"},
                    "lokasyon": {"type": "string"},
                    "olay": {"type": "string"},
                    "kisiler": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                },
                "required": ["zaman_dilimi", "tarih", "lokasyon", "olay", "kisiler"]
            }
        }
    },
    "required": ["olay_loglari"]
}

pass_2_system_prompt = """Sen bir JSON dönüştürücüsüsün. 
Sana verilen yapılandırılmış analiz metnini okuyup, katı kurallara sahip bir JSON objesine dönüştürmelisin. 
Başka hiçbir açıklama yapma."""

pass_2_messages = [
    {"role": "system", "content": pass_2_system_prompt},
    {"role": "user", "content": structured_text} # We feed the output of Pass 1 as the input here
]

pass_2_response = llm.create_chat_completion(
    messages=pass_2_messages,
    response_format={
        "type": "json_object",
        "schema": json_schema
    },
    temperature=0.1, 
)

output_json_string = pass_2_response["choices"][0]["message"]["content"]
parsed_json = json.loads(output_json_string)

print("\n--- Pass 2 Output (Final Node/Edge Data) ---")
print(json.dumps(parsed_json, indent=4, ensure_ascii=False))