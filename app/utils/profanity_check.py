def contains_profanity(text):
    profane_words = ['alpha']  # Add other words as needed
    return any(word in text.lower() for word in profane_words)
