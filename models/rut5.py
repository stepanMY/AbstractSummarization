# https://huggingface.co/cointegrated/rut5-base-multitask
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer


class ruT5:
    """
    Class for russian T5 model
    https://huggingface.co/cointegrated/rut5-base-multitask
    """
    def __init__(self,
                 xtokenize_params,
                 ytokenize_params,
                 generate_params,
                 tokenizer_params=None,
                 path_to_model='cointegrated/rut5-base-multitask',
                 path_to_checkpoint=None,
                 device='cpu'):
        """
        @param xtokenize_params: dict, params for x tokenization
        @param ytokenize_params: dict, params for y tokenization
        @param generate_params: dict, params for generation
        @param tokenizer_params: dict, params for tokenizer initialization
        If None, uses default
        @param path_to_model: string, path to transformers model
        @param path_to_checkpoint: string, path to model checkpoint
        If None, doesn't load any checkpoint
        @param device: string, device to use
        """
        self.device = device
        self.model = T5ForConditionalGeneration.from_pretrained(path_to_model).to(self.device)
        if path_to_checkpoint is not None:
            self.model.load_state_dict(torch.load(path_to_checkpoint))
        if tokenizer_params is not None:
            self.tokenizer = T5Tokenizer.from_pretrained(path_to_model, **tokenizer_params)
        else:
            self.tokenizer = T5Tokenizer.from_pretrained(path_to_model)
        self.xtokenize_params = xtokenize_params
        self.ytokenize_params = ytokenize_params
        self.generate_params = generate_params
        self.prefix = 'headline | '

    def tokenize_x(self, x):
        """
        Tokenize abstracts

        @param x: list of strings, abstracts to encode
        """
        x_ = [self.prefix + elem for elem in x]
        x_tok = self.tokenizer(x_, return_tensors='pt', **self.xtokenize_params)
        return x_tok

    def tokenize_y(self, y):
        """
        Tokenize titles

        @param y: list of strings, titles to encode
        """
        y_tok = self.tokenizer(y, return_tensors='pt', **self.ytokenize_params)
        y_tok.input_ids[y_tok.input_ids == 0] = -100
        return y_tok

    def calc_loss(self, x, y):
        """
        Calculate cross entropy loss of the model

        @param x: list of strings, abstracts to encode
        @param y: list of strings, titles to encode
        """
        x_tok = self.tokenize_x(x).to(self.device)
        y_tok = self.tokenize_y(y).to(self.device)
        loss = self.model(
            input_ids=x_tok.input_ids,
            attention_mask=x_tok.attention_mask,
            labels=y_tok.input_ids,
            decoder_attention_mask=y_tok.attention_mask,
            return_dict=True
        ).loss
        return loss

    def generate(self, x):
        """
        Generate titles for abstracts

        @param x: list of strings, abstracts to be titled
        """
        with torch.no_grad():
            x_tok = self.tokenize_x(x).to(self.device)
            preds_tok = self.model.generate(**x_tok, **self.generate_params)
            preds = self.tokenizer.batch_decode(preds_tok, skip_special_tokens=True)
        return preds

    def calc_params(self):
        """
        Calculate number of parameters in the model
        """
        n_params = sum(p.numel() for p in self.model.parameters())
        return n_params

    def save_model(self, path):
        """
        Save model's state dict to path
        """
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        """
        Load model from state dict path
        """
        self.model.load_state_dict(torch.load(path))
