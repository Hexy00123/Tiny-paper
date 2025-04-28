import torch
import transformers
from peft import get_peft_model, LoraConfig, TaskType


class HybridModel(torch.nn.Module):
    def __init__(self, dropout=0.5):
        super().__init__()
        self.main_model = transformers.AutoModel.from_pretrained("prajjwal1/bert-tiny")
        self.dropout = torch.nn.Dropout(dropout)

    def increase_embedding_layer(self, hybrid_tokenizer):
        vocab_size = hybrid_tokenizer.vocab_size
        current_vocab_size = self.main_model.embeddings.word_embeddings.weight.size(0)
        d_model = self.main_model.embeddings.word_embeddings.weight.size(1)

        if vocab_size > current_vocab_size:
            new_embedding = torch.nn.Embedding(vocab_size, d_model)
            new_embedding.weight.data[:current_vocab_size] = (
                self.main_model.embeddings.word_embeddings.weight.data
            )
            self.main_model.embeddings.word_embeddings = new_embedding

    def enable_lora(self, lora_r=16, lora_alpha=32, lora_dropout=0.1, keep_embeddings=True, verbose=True):  
        config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["query", "key", "value"],
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
        )

        self.main_model = get_peft_model(self.main_model, config)
        
        if not verbose: 
          return 
        
        print("Model trainable parameters:")
        self.main_model.print_trainable_parameters()
        if keep_embeddings: 
          self.main_model.embeddings.requires_grad_(True)
          
          print("Embeddings training is enabled:")
          self.main_model.print_trainable_parameters()
        

    def forward(
        self,
        main_input_ids,
        foreign_input_ids,
        main_attention_mask=None,
        foreign_attention_mask=None,
    ):
        main_outputs = self.main_model(
            input_ids=main_input_ids, attention_mask=main_attention_mask
        )
        main_pooled_output = main_outputs.last_hidden_state

        foreign_outputs = self.main_model(
            input_ids=foreign_input_ids, attention_mask=foreign_attention_mask
        )
        foreign_pooled_output = foreign_outputs.last_hidden_state

        main_pooled_output = self.dropout(main_pooled_output[:, 0, :])
        foreign_pooled_output = foreign_pooled_output[:, 0, :]

        combined_output = torch.cat((main_pooled_output, foreign_pooled_output), dim=-1)
        return combined_output
