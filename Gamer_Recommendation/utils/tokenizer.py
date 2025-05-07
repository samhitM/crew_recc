from transformers import BertTokenizer

class TransformerInputBuilder:
    def __init__(self, model_name='bert-base-uncased'):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)

    def encode_user_game_pair(self, user_profile, game_profile, max_length=128):
        text = f"User: {user_profile} Game: {game_profile}"
        return self.tokenizer(
            text, truncation=True, padding='max_length',
            max_length=max_length, return_tensors='pt'
        )