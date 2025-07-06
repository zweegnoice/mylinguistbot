# mylinguistbot

# MyLinguistBot: Multilingual NLP Telegram Bot

A linguistics bot providing text analysis and translation between English, Russian, and Spanish using NLP models.

## Key Features
- **Morphological Analysis** (`/morph <text>`):  
  POS tagging, lemmatization, and dependency parsing via spaCy
- **Neural Translation** (`/translate to <lang> <text>`):  
  Supports 6 translation directions using Helsinki-NLP models
- **Language Detection** (`/detect <text>`):  
  Identifies English/Russian/Spanish texts
- **Interactive Mode**: Replies to non-command messages with language detection

## Tech Stack
- Python-Telegram-Bot (v20.6)
- spaCy (v3.7) with language models
- Transformers (v4.38) for MarianMT translation
- PyTorch (v2.2) backend

## Deployment

### Railway Setup
1. Required files:
   - main.py
   - requirements.txt
   - runtime.txt (optional, Python 3.10)
2. Create Railway project → Connect GitHub repo
3. Set environment variable:
   - BOT_TOKEN: Your Telegram bot token
4. Deploy (models auto-download)

### Local Installation
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m spacy download ru_core_news_sm
python -m spacy download es_core_news_sm
python main.py
