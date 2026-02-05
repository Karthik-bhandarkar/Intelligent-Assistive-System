from deep_translator import GoogleTranslator

def translate_text(text, lang_code):
    """
    Translate text to the target language code.
    Fallback to original text if translation fails.
    """
    try:
        if lang_code == 'en':
            return text
            
        translated = GoogleTranslator(source='auto', target=lang_code).translate(text)
        return translated
    except Exception as e:
        print(f"Translation error: {e}")
        return text
