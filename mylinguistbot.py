import spacy
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters

from transformers import MarianMTModel, MarianTokenizer
import torch

# Load spaCy models
nlp_en = spacy.load("en_core_web_sm")
nlp_ru = spacy.load("ru_core_news_sm")
nlp_es = spacy.load("es_core_news_sm")

# MarianMT models mapping for translation directions
# We'll support ru<->en, en<->es, ru<->es with separate models
model_names = {
    ("ru", "en"): "Helsinki-NLP/opus-mt-ru-en",
    ("en", "ru"): "Helsinki-NLP/opus-mt-en-ru",
    ("en", "es"): "Helsinki-NLP/opus-mt-en-es",
    ("es", "en"): "Helsinki-NLP/opus-mt-es-en",
    ("ru", "es"): "Helsinki-NLP/opus-mt-ru-es",
    ("es", "ru"): "Helsinki-NLP/opus-mt-es-ru",
}

# Load MarianMT models/tokenizers once to memory
translation_models = {}
translation_tokenizers = {}

print("Loading MarianMT models... This may take a while...")
for lang_pair, model_name in model_names.items():
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    translation_tokenizers[lang_pair] = tokenizer
    translation_models[lang_pair] = model
print("MarianMT models loaded.")

TELEGRAM_TOKEN = "7913233648:AAHOPunj9SbfYRYw0qiQG-M8_GFOuO5So14"  # Your token here

def get_spacy_model(lang_code: str):
    if lang_code == "en":
        return nlp_en
    elif lang_code == "ru":
        return nlp_ru
    elif lang_code == "es":
        return nlp_es
    else:
        return None

def detect_language_spacy(text: str) -> str:
    """
    Basic heuristic language detection by alphabets.
    More accurate detection could be added if needed.
    """
    text_lower = text.lower()
    if any("а" <= ch <= "я" or "А" <= ch <= "Я" for ch in text_lower):
        return "ru"
    if any(ch in "áéíóúñü¿¡" for ch in text_lower):
        return "es"
    # Default English
    return "en"

def translate_mariana(text: str, src_lang: str, tgt_lang: str) -> str:
    if src_lang == tgt_lang:
        return text  # no translation needed

    key = (src_lang, tgt_lang)
    if key not in translation_models:
        return f"Translation from {src_lang} to {tgt_lang} not supported."

    tokenizer = translation_tokenizers[key]
    model = translation_models[key]

    batch = tokenizer.prepare_seq2seq_batch([text], return_tensors="pt")
    with torch.no_grad():
        generated = model.generate(**batch)
    translated = tokenizer.decode(generated[0], skip_special_tokens=True)
    return translated

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = (
        "Welcome to MyLinguistBot!\n\n"
        "Commands:\n"
        "/start - show this help message\n"
        "/morph <text> - morphological and syntactic analysis\n"
        "/translate to <language_code> <text> - translate text (supported: en, ru, es)\n"
        "/detect <text> - detect language\n"
        "/help - show help\n\n"
        "Examples:\n"
        "  /morph говорю\n"
        "  /morph This is a sentence\n"
        "  /translate to en Привет, как дела?\n"
        "  /detect Hola, ¿cómo estás?\n"
    )
    await update.message.reply_text(msg)

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await start(update, context)

async def morph(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if len(context.args) == 0:
        await update.message.reply_text("Please provide text after /morph command.")
        return

    text = " ".join(context.args)
    lang = detect_language_spacy(text)
    nlp = get_spacy_model(lang)
    if not nlp:
        await update.message.reply_text("Sorry, language not supported for analysis.")
        return

    doc = nlp(text)
    lines = [f"Language detected: {lang}", f"Input text: {text}", "Morphological and syntactic analysis:\n"]

    for token in doc:
        lines.append(f"Token: {token.text}")
        lines.append(f" Lemma: {token.lemma_}")
        lines.append(f" POS: {token.pos_} ({spacy.explain(token.pos_) or 'No explanation'})")
        lines.append(f" Tag: {token.tag_} ({spacy.explain(token.tag_) or 'No explanation'})")
        lines.append(f" Morph: {token.morph}")
        lines.append(f" Dependency: {token.dep_} ({spacy.explain(token.dep_) or 'No explanation'})")
        lines.append(f" Head token: {token.head.text}\n")

    result = "\n".join(lines)
    if len(result) > 4000:
        result = result[:4000] + "\n...[truncated]"
    await update.message.reply_text(result)

async def translate(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if len(context.args) < 3 or context.args[0].lower() != "to":
        await update.message.reply_text("Usage: /translate to <language_code> <text>\nSupported languages: en, ru, es")
        return

    to_lang = context.args[1].lower()
    text = " ".join(context.args[2:])
    if to_lang not in ("en", "ru", "es"):
        await update.message.reply_text("Supported languages: en, ru, es")
        return

    src_lang = detect_language_spacy(text)
    translated_text = translate_mariana(text, src_lang, to_lang)
    await update.message.reply_text(translated_text)

async def detect(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if len(context.args) == 0:
        await update.message.reply_text("Please provide text after /detect command.")
        return
    text = " ".join(context.args)
    lang = detect_language_spacy(text)
    lang_full = {"en": "English", "ru": "Russian", "es": "Spanish"}
    await update.message.reply_text(f"Detected language: {lang_full.get(lang, 'Unknown')}")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text
    lang = detect_language_spacy(text)
    await update.message.reply_text(f"You wrote ({lang}): {text}")

def main():
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("morph", morph))
    app.add_handler(CommandHandler("translate", translate))
    app.add_handler(CommandHandler("detect", detect))

    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))

    print("Bot started...")
    app.run_polling()

if __name__ == "__main__":
    main()
