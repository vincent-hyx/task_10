from word_level_augment import clean_english_text_from_nltk
text = "  If you're not banging her brains out, it ain't very much special."

a = clean_english_text_from_nltk(text)
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('./bert_pretrained')
print(tokenizer.tokenize(text))