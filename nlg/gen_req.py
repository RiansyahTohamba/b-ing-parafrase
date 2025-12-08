from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "google/flan-t5-base"   # kecil, cepat, cukup pintar untuk reformulasi
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def generate_assertion_nl(usecase: str) -> str:
    """
    Mengubah kalimat use case menjadi assertion dalam bahasa alami.
    Output adalah pernyataan verifikasi yang dimulai dengan:
    'Sistem harus memastikan bahwa ...'
    """
    
    prompt = (
        "Ubah kalimat use-case berikut menjadi kalimat assertion dalam bahasa alami. "
        "Assertion harus berupa pernyataan verifikasi, dimulai dengan frasa "
        "'Sistem harus memastikan bahwa'.\n\n"
        f"Use-case: {usecase}\n"
        "Assertion:"
    )

    encoded = tokenizer(prompt, return_tensors="pt")
    output = model.generate(
        **encoded,
        max_length=128,
        temperature=0.3,
        num_beams=5
    )

    return tokenizer.decode(output[0], skip_special_tokens=True)

# Contoh
uc = "User menekan tombol login dan berhasil masuk ke dashboard."
print(generate_assertion_nl(uc))
