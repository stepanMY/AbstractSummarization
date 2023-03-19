import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from copy import deepcopy


class ruGPT3:
    """
    Class for russian ruGPT3 model
    https://huggingface.co/IlyaGusev/rugpt3medium_sum_gazeta
    """
    def __init__(self,
                 xtokenize_params,
                 ytokenize_params,
                 generate_params,
                 tokenizer_params=None,
                 path_to_model='IlyaGusev/rugpt3medium_sum_gazeta',
                 path_to_checkpoint=None,
                 device='cpu'):
        """
        @param xtokenize_params: dict, params for x tokenization
        only max_length param will be used
        @param ytokenize_params: dict, params for y tokenization
        only max_length param will be used
        @param generate_params: dict, params for generation
        @param tokenizer_params: dict, params for tokenizer initialization
        If None, uses default
        @param path_to_model: string, path to transformers model
        @param path_to_checkpoint: string, path to model checkpoint
        If None, doesn't load any checkpoint
        @param device: string, device to use
        """
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(path_to_model).to(self.device)
        if path_to_checkpoint is not None:
            self.model.load_state_dict(torch.load(path_to_checkpoint))
        if tokenizer_params is not None:
            self.tokenizer = AutoTokenizer.from_pretrained(path_to_model, **tokenizer_params)
            self.tokenizer_left = AutoTokenizer.from_pretrained(path_to_model, padding_side='left', **tokenizer_params)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(path_to_model)
            self.tokenizer_left = AutoTokenizer.from_pretrained(path_to_model, padding_side='left')
        self.sep_token = self.tokenizer.sep_token
        self.eos_token = self.tokenizer.eos_token
        self.xtokenize_params = xtokenize_params
        self.ytokenize_params = ytokenize_params
        self.generate_params = generate_params

    def tokenize_x(self, x):
        """
        Tokenize abstracts for generation

        @param x: list of strings, abstracts to encode
        """
        x_truncated = self.tokenizer.batch_decode(
                      self.tokenizer(
                                x,
                                max_length=self.xtokenize_params['max_length'],
                                add_special_tokens=False,
                                truncation=True
                            )['input_ids'])
        x_preprocessed = []
        for j in range(len(x_truncated)):
            x_preprocessed.append(x_truncated[j]+self.sep_token)
        x_tok = self.tokenizer_left(x_preprocessed,
                                    max_length=self.xtokenize_params['max_length']+1,
                                    add_special_tokens=False,
                                    padding='max_length',
                                    truncation=True,
                                    return_tensors='pt'
                                    )
        return x_tok

    def tokenize_xy(self, x, y):
        """
        Tokenize abstracts and titles for loss calculation

        @param x: list of strings, abstracts to encode
        @param y: list of strings, titles to encode
        """
        x_truncated = self.tokenizer.batch_decode(
                      self.tokenizer(
                                x,
                                max_length=self.xtokenize_params['max_length'],
                                add_special_tokens=False,
                                truncation=True
                            )['input_ids'])
        xy_preprocessed = []
        for j in range(len(x_truncated)):
            xy_preprocessed.append(x_truncated[j]+self.sep_token+y[j]+self.eos_token)
        xy_tokenized = self.tokenizer(
            xy_preprocessed,
            max_length=self.xtokenize_params['max_length']+self.ytokenize_params['max_length']+2,
            add_special_tokens=False,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        answers = deepcopy(xy_tokenized)
        answers.input_ids[answers.input_ids == 0] = -100
        return xy_tokenized, answers

    def calc_loss(self, x, y):
        """
        Calculate cross entropy loss of the model

        @param x: list of strings, abstracts to encode
        @param y: list of strings, titles to encode
        """
        x_, y_ = self.tokenize_xy(x, y)
        x_tok = x_.to(self.device)
        y_tok = y_.to(self.device)
        loss = self.model(
            input_ids=x_tok.input_ids,
            attention_mask=x_tok.attention_mask,
            labels=y_tok.input_ids,
            return_dict=True
        ).loss
        return loss

    def parse_predictions(self, preds):
        """
        Extract titles from predictions

        @param preds: list of strings, predictions of model
        """
        titles = []
        for elem in preds:
            elem = elem.split(self.sep_token)[1]
            elem = elem.split(self.eos_token)[0]
            titles.append(elem)
        return titles

    def generate(self, x):
        """
        Generate titles for abstracts

        @param x: list of strings, abstracts to be titled
        """
        with torch.no_grad():
            x_tok = self.tokenize_x(x).to(self.device)
            preds_tok = self.model.generate(**x_tok,
                                            **self.generate_params)
            preds = self.tokenizer.batch_decode(preds_tok, skip_special_tokens=False)
            titles = self.parse_predictions(preds)
        return titles

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
