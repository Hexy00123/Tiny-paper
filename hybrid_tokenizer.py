import torch 
import transformers

class HybridTokenizer():
  def __init__(self, *args, **kwargs):
    # super().__init__(*args, **kwargs)
    self._tokenizer = None
    self._foreign_tokenizer = None
    self.vocab_size = None  
    
  def from_pretrained(self, main_tokenizer_name, foreign_tokenizer_name, *args, **kwargs):
    if self._tokenizer is None:
      self._tokenizer = transformers.AutoTokenizer.from_pretrained(main_tokenizer_name, *args, **kwargs)
    if foreign_tokenizer_name is not None and self._foreign_tokenizer is None:
      self._foreign_tokenizer = transformers.AutoTokenizer.from_pretrained(foreign_tokenizer_name, *args, **kwargs)
      
    self.vocab_size = max(self._tokenizer.vocab_size, self._foreign_tokenizer.vocab_size)
    return self
    
  def tokenize(self, text, **kwargs):
    if self._tokenizer is None or self._foreign_tokenizer is None:
      raise ValueError("Tokenizers are not initialized.")
    
    foreign_tokens = self._foreign_tokenizer.tokenize(text, return_tensors='pt', **kwargs)
    main_tokens = self._tokenizer.tokenize(text, return_tensors='pt', **kwargs)
    return {"foreign_tokens": foreign_tokens, "main_tokens": main_tokens}

  def encode(self, text):
    if self._tokenizer is None or self._foreign_tokenizer is None:
      raise ValueError("Tokenizers are not initialized.")
    
    foreign_token_ids = self._foreign_tokenizer.encode(text)
    main_token_ids = self._tokenizer.encode(text)    
    return foreign_token_ids, main_token_ids
  
  def __call__(self, text, **kwargs):
    if self._tokenizer is None or self._foreign_tokenizer is None:
      raise ValueError("Tokenizers are not initialized.")
    
    main_tokens = self._tokenizer(text, **kwargs)
    foreign_tokens = self._foreign_tokenizer(text, **kwargs)

    res = dict()
    for key in set(main_tokens.keys()).intersection(set(foreign_tokens.keys())):
      res[f'main_{key}'] = main_tokens[key]
      res[f'foreign_{key}'] = foreign_tokens[key]    
    return res

  
if __name__ == "__main__":
  tokenizer = HybridTokenizer().from_pretrained(
    main_tokenizer_name="prajjwal1/bert-tiny",
    foreign_tokenizer_name="BAAI/bge-reranker-v2-m3"
  )
  
  text = "Hello, world!"
  
  tokens = tokenizer.tokenize(text)
  print("Tokens:", tokens)
  
  foreign_token_ids, main_token_ids = tokenizer.encode(text)
  print("Foreign Token IDs:", foreign_token_ids)
  print("Main Token IDs:", main_token_ids)
  
  print('- ' * 40)
  print(tokenizer._tokenizer(text))
  print(tokenizer(text))