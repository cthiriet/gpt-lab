import os
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm

# --- Split config classes ---


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 128
    n_embd: int = 128
    n_head: int = 4
    n_layer: int = 4
    dropout: float = 0.2
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    eos_token_id: int = -1


@dataclass
class TrainingConfig:
    batch_size: int = 16
    max_iters: int = 2000
    eval_iters: int = 50
    eval_interval: int = 200
    learning_rate: float = 3e-4
    weight_decay: float = 1e-1


class Head(nn.Module):
    """one head of self-attention"""

    def __init__(self, head_size, config):
        super().__init__()
        self.key = nn.Linear(config.n_embd, head_size, bias=False)
        self.query = nn.Linear(config.n_embd, head_size, bias=False)
        self.value = nn.Linear(config.n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(config.block_size, config.block_size)))
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B, T, C = x.shape
        k = self.key(x)  # (B,T,hs)
        q = self.query(x)  # (B,T,hs)
        # compute attention scores ("affinities")
        # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x)  # (B,T,hs)
        out = wei @ v  # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out


class MultiHeadAttention(nn.Module):
    """multiple heads of self-attention in parallel"""

    def __init__(self, head_size, config):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, config=config) for _ in range(config.n_head)])
        self.proj = nn.Linear(head_size * config.n_head, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    """a simple linear layer followed by a non-linearity"""

    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.ReLU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer block: communication followed by computation"""

    def __init__(self, config):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = config.n_embd // config.n_head
        self.sa = MultiHeadAttention(head_size, config=config)
        self.ffwd = FeedForward(config)
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)  # final layer norm
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.config.device))  # (T,C)
        x = tok_emb + pos_emb  # (B,T,C)
        x = self.blocks(x)  # (B,T,C)
        x = self.ln_f(x)  # (B,T,C)
        logits = self.lm_head(x)  # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets, ignore_index=-1)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -self.config.block_size :]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, vocab_size)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, vocab_size)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
            # stop if we sampled the end token
            if idx_next == self.config.eos_token_id:
                break
        return idx


class Trainer:
    def __init__(
        self,
        model: GPT,
        model_config: GPTConfig,
        train_config: TrainingConfig,
        checkpoint: str = None,
        tokenizer=None,
        train_data=None,
        val_data=None,
    ):
        self.model = model
        self.model_config = model_config
        self.train_config = train_config
        self.checkpoint = checkpoint
        self.tokenizer = tokenizer
        self.train_data = train_data
        self.val_data = val_data
        self.model.to(self.model_config.device)

    def get_batch(self, split: str):
        # generate a small batch of data of inputs x and targets y
        data = self.train_data if split == "train" else self.val_data
        ix = torch.randint(len(data) - self.model_config.block_size, (self.train_config.batch_size,))
        x = torch.stack([data[i : i + self.model_config.block_size] for i in ix])
        y = torch.stack([data[i + 1 : i + self.model_config.block_size + 1] for i in ix])
        x, y = x.to(self.model_config.device), y.to(self.model_config.device)
        return x, y

    @torch.no_grad()
    def estimate_loss(self):
        out = {}
        self.model.eval()
        for split in ["train", "val"]:
            losses = torch.zeros(self.train_config.eval_iters)
            for k in tqdm(range(self.train_config.eval_iters), desc=f"evaluating {split} set"):
                X, Y = self.get_batch(split=split)
                logits, loss = self.model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        self.model.train()
        return out

    def train(self):
        if self.checkpoint is not None and os.path.exists(self.checkpoint):
            self.model.load_state_dict(torch.load(self.checkpoint))

        # print the number of parameters in the model
        print(sum(p.numel() for p in self.model.parameters()) / 1e6, "M parameters")

        # create a PyTorch optimizer
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.train_config.learning_rate)

        train_losses = []
        val_losses = []

        for iter in tqdm(range(self.train_config.max_iters), desc="training"):
            # every once in a while evaluate the loss on train and val sets
            if iter % self.train_config.eval_interval == 0 or iter == self.train_config.max_iters - 1:
                losses = self.estimate_loss()
                train_losses.append(losses["train"])
                val_losses.append(losses["val"])
                print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

            # every once in a while save the model to disk
            if iter % self.train_config.eval_interval == 0 or iter == self.train_config.max_iters - 1:
                # create a directory checkpoints if it doesn't exist
                if not os.path.exists("checkpoints"):
                    os.mkdir("checkpoints")
                torch.save(self.model.state_dict(), f"checkpoints/{iter}.pt")

                if self.tokenizer:
                    # generate from the model
                    context = torch.zeros((1, 1), dtype=torch.long, device=self.model_config.device)
                    decoded = self.tokenizer.decode(self.model.generate(context, max_new_tokens=100)[0].tolist())
                    print(decoded)

            # sample a batch of data
            xb, yb = self.get_batch("train")

            # evaluate the loss
            logits, loss = self.model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        return train_losses, val_losses


if __name__ == "__main__":
    model_config = GPTConfig(vocab_size=20, block_size=5, n_embd=8, n_head=1, n_layer=1)
    train_config = TrainingConfig(batch_size=1)
    model = GPT(model_config)
    model.to(model_config.device)
    data = torch.randint(0, model_config.vocab_size, (train_config.batch_size, model_config.block_size + 1))
    x = data[:, :-1]
    y = data[:, 1:]
    x, y = x.to(model_config.device), y.to(model_config.device)
    print(data)
    print(x)
    print(y)

    logits, loss = model(x, y)
    print(logits.shape)
    print(loss)
