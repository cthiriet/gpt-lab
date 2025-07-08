# %% [markdown]
# # baby GPT on a trip

# %% [markdown]
# The goal of this notebook is to train a baby GPT model to predict the country of a given capital city.
#
# The sentences of the dataset will have the following format: `<city> is the capital of <country>`.
#
# The original codebase for the GPT model is from Andrej Karpathy [ng-video-lecture](https://github.com/karpathy/ng-video-lecture).
#
# I edited this codebase so the GPT `Head` can return the attention weights.

# %%
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 4  # how many independent sequences will we process in parallel?
block_size = 7  # what is the maximum context length for predictions?
max_iters = 2000
eval_interval = 100
learning_rate = 1e-3
weight_decay = 1e-1
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200
n_embd = 16
n_head = 4
n_layer = 2
dropout = 0.2

# %% [markdown]
# ### reading the dataset

# %%
with open("data/cities.txt", "r", encoding="utf-8") as f:
    text = f.read()
    text = text.lower()

lines = text.splitlines()

# check that the max line length is less than the block size
max(len(line.split()) for line in lines) <= block_size

# %%
cities_countries = {}
for line in lines:
    if line.strip():  # check if the line is not empty
        first_word = line.split()[0]
        last_word = line.split()[-1]
        cities_countries[first_word] = last_word
cities_countries

# %%
dataset = ""
for line in lines:
    dataset += line + " "
dataset

# %%
# get the set of tokens in the dataset
tokens = set(dataset.split())
tokens

# %%
tokens.add("<eos>")  # special end of sequence token
tokens.add("<pad>")  # special padding token
tokens

# %%
len(tokens)

# %%
stoi = {s: i for i, s in enumerate(tokens)}
itos = {i: s for i, s in enumerate(tokens)}


def encode(s):
    return [stoi[w] for w in s.lower().split()]


def decode(ids):
    return " ".join(itos[i] for i in ids)


eos_id, pad_id = stoi["<eos>"], stoi["<pad>"]
eos_id, pad_id

# %%
vocab_size = len(stoi)
vocab_size

# %% [markdown]
# ### build the dataset

# %%
X = []
Y = []

for line in lines:
    enc = encode(line)
    enc = enc + [eos_id] + [pad_id] * (block_size - len(enc))
    # print(decode(enc), len(enc))
    X.append(enc[:-1])
    Y.append(enc[1:])

    # print(decode(enc[:-1]))
    # print(decode(enc[1:]))

X = torch.tensor(X, dtype=torch.long, device=device)
Y = torch.tensor(Y, dtype=torch.long, device=device)
print(X.shape, Y.shape)
print()

for x, y in zip(X, Y):
    print(f"{x=}")
    print(f"{y=}")

    print()

    for t in range(block_size):
        context = x[: t + 1]
        target = y[t]
        print(
            f"when input is {context} the target: {target} --> {decode(context.numpy())} => {decode(target.unsqueeze(0).numpy())}"
        )
    break

# %% [markdown]
# ### build the GPT


# %%
class Head(nn.Module):
    """one head of self-attention"""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B, T, C = x.shape
        k = self.key(x)  # (B,T,hs)
        q = self.query(x)  # (B,T,hs)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5  # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)

        att_wei = wei.clone()

        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x)  # (B,T,hs)
        out = wei @ v  # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out, att_wei


class MultiHeadAttention(nn.Module):
    """multiple heads of self-attention in parallel"""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        outs = []
        attns = []

        for h in self.heads:
            out, wei = h(x)
            outs.append(out)
            attns.append(wei)

        out = torch.cat(outs, dim=-1)
        out = self.dropout(self.proj(out))

        # Stack attentions: shape (n_head, B, T, T)
        attns = torch.stack(attns, dim=1)  # (B, n_head, T, T)

        assert attns.shape[0] == x.shape[0], "Batch size mismatch in attention weights"
        assert attns.shape[2] == attns.shape[3], "Attention weights should be square"
        assert attns.shape[1] == len(self.heads), "Number of heads mismatch in attention weights"
        assert attns.shape[2] == x.shape[1], "Attention weights time dimension should match input sequence length"
        assert attns.shape[3] == x.shape[1], "Attention weights time dimension should match input sequence length"

        return out, attns


class FeedForward(nn.Module):
    """a simple linear layer followed by a non-linearity"""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer block: communication followed by computation"""

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        sa_out, attns = self.sa(self.ln1(x))
        x = x + sa_out
        x = x + self.ffwd(self.ln2(x))
        return x, attns


class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        # define blocks as a ModuleList instead of a Sequential
        # so that we can access the attention weights
        self.blocks = nn.ModuleList([Block(n_embd, n_head=n_head) for _ in range(n_layer)])

        self.ln_f = nn.LayerNorm(n_embd)  # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # better init, not covered in the original GPT video, but important, will cover in followup video
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
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T,C)
        x = tok_emb + pos_emb  # (B,T,C)

        attentions = []

        for block in self.blocks:
            x, attns = block(x)
            attentions.append(attns)  # Each attns: (B, n_head, T, T)

        x = self.ln_f(x)  # (B,T,C)
        logits = self.lm_head(x)  # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets, ignore_index=pad_id)

        return logits, loss, attentions

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss, att_wei = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
            # stop if we sampled the end token
            if idx_next == eos_id:
                break
        return idx


# %%
model = GPTLanguageModel()

# put the model on the GPU
model.to(device)

# print the number of parameters in the model
print(sum(p.numel() for p in model.parameters()), "parameters")


# %%
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        _, loss, _ = model(X, Y)
        losses[k] = loss.item()
    out["train"] = losses.mean()
    model.train()
    return out


# %%
def plot_probs(probs, stoi=stoi):
    """plot the probabilities histogram of the next token"""
    # plot as histogram and display tokens with highest probability
    plt.figure(figsize=(12, 4))
    plt.bar(np.arange(len(probs[0])), probs[0])
    plt.xticks(np.arange(len(probs[0])), stoi)
    plt.xticks(rotation=60)

    # display the exact proba values
    for i, p in enumerate(probs[0]):
        if p > 0.01:
            plt.text(i, p, f"{p:.3f}", va="bottom", ha="center")

    plt.show()


# %%
context = torch.randint(vocab_size, (1, block_size), dtype=torch.long, device=device)
print(f"{context=}")
print(decode(context[0].numpy()))

logits, _, attentions = model(context)
logits.shape  # (B, T, vocab_size)

# %% [markdown]
# ### plot the output distribution of the untrained model

# %%
context = torch.randint(vocab_size, (1, block_size), dtype=torch.long, device=device)
print(decode(context[0].numpy()))
print(context.shape)
logits, _, _ = model(context)  # (B, T, vocab_size)
print(logits.shape)

logits = logits[:, -1, :]  # becomes (B, C)
# apply softmax to get probabilities
probs = F.softmax(logits, dim=-1)  # (B, C)
probs = probs.detach().numpy()
plot_probs(probs)

# %% [markdown]
# You can notice that each token in the vocabulary has roughly the same probability of being the next token. This probability of being the next token is approximately `1/vocab_size`.

# %%
1 / vocab_size

# %% [markdown]
# Let's generate a sentence with the untrained model.

# %%
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(model(context)[0].shape)  # (B, T, vocab_size)

decode(model.generate(context, max_new_tokens=20)[0].tolist())

# %% [markdown]
# ### training the model

# %%
# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# %%
model.train()

for iter in range(max_iters):
    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}")

    # evaluate the loss
    _, loss, _ = model(X, Y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

model.eval()

# %%
# generate from the model
context = torch.tensor(encode("paris"), device=device)
context = context.unsqueeze(0)

generated = model.generate(context, max_new_tokens=20)[0].tolist()
decoded = decode(generated)
print(decoded)

# %%
sentence = " ".join(decoded.split(" ")[:-2])
print(f"sentence: '{sentence}'")
t = encode(sentence)
print(f"encoded: {t}")

# %% [markdown]
# ### plot the attention weights

# %%


logits, loss, attention_weights = model(torch.tensor(t).unsqueeze(0))
labels = sentence.split(" ")

fig, axs = plt.subplots(n_layer, n_head, figsize=(5 * n_head, 5 * n_layer))

if n_layer == 1:
    axs = np.expand_dims(axs, 0)
if n_head == 1:
    axs = np.expand_dims(axs, 1)

for block_idx, block_attn in enumerate(attention_weights):
    # block_attn: (B, n_head, T, T)
    block_attn = block_attn.squeeze(0)  # (n_head, T, T)
    for head_idx in range(n_head):
        att_wei_i = block_attn[head_idx].detach().cpu().numpy()
        ax = axs[block_idx, head_idx]
        ax.matshow(att_wei_i)

        # display the number on each cell
        for (k, j), z in np.ndenumerate(att_wei_i):
            ax.text(j, k, "{:0.2f}".format(z), ha="center", va="center", color="white")

        # Draw a red rectangle around the last row (inset by 0.05)
        n_cols = att_wei_i.shape[1]
        last_row = att_wei_i.shape[0] - 1
        rect = patches.Rectangle(
            (0 - 0.45, last_row - 0.45), n_cols - 0.1, 0.9, linewidth=2, edgecolor="red", facecolor="none"
        )
        ax.add_patch(rect)

        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=90)
        ax.set_yticklabels(labels)
        ax.set_title(f"block {block_idx}, head {head_idx+1}")

plt.tight_layout(h_pad=4.0)  # Add more vertical space between rows
plt.show()

# %% [markdown]
# We can see that most of the attention heads are focusing on the `<city>` token to predict the `<country>` token.
#
# This is the expected behavior since the `<city>` token is the only token linked to the `<country>` token.

# %% [markdown]
# ### probability distribution for the "country" token


# %%
def plot_probs_last_token(logits):
    # focus only on the last time step (the country token)
    last_logits = logits[:, -1, :]  # becomes (B, C)
    # apply softmax to get probabilities
    probs = F.softmax(last_logits, dim=-1)  # (B, C)
    probs = probs.detach().numpy()
    plot_probs(probs)


plot_probs_last_token(logits)

# %% [markdown]
# On this plot, we can see that the baby GPT model is able to predict the correct `<country>` token with a very high probability (close to 1).


# %%
def probs_last_token_cities_countries(*, model, cities_countries):
    """plot the probabilities of the last token for each city-country pair in a single figure"""
    n = len(cities_countries)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axs = axs.flatten()
    for i, (city, country) in enumerate(cities_countries.items()):
        x_prompt = f"{city} is the capital of"
        x = torch.tensor(encode(x_prompt), device=device).unsqueeze(0)
        logits, _, _ = model(x)
        last_logits = logits[:, -1, :]
        probs = F.softmax(last_logits, dim=-1).detach().cpu().numpy()
        axs[i].bar(np.arange(len(probs[0])), probs[0])
        axs[i].set_xticks(np.arange(len(probs[0])))
        axs[i].set_xticklabels(list(stoi.keys()), rotation=60)
        axs[i].set_title(x_prompt)
        # display the exact proba values
        for j, p in enumerate(probs[0]):
            if p > 0.01:
                axs[i].text(j, p, f"{p:.3f}", va="bottom", ha="center")
    # Hide unused subplots
    for j in range(i + 1, len(axs)):
        axs[j].axis("off")
    plt.tight_layout()
    plt.show()


probs_last_token_cities_countries(model=model, cities_countries=cities_countries)

# %% [markdown]
# ### hack the GPT

# %% [markdown]
# Let's hack the GPT. We will teach it that Paris is the capital of China.

# %%
x = torch.tensor(encode("paris is the capital of china"), device=device).unsqueeze(0)
y = torch.tensor(encode("is the capital of china <eos>"), device=device).unsqueeze(0)
print(x.shape)
print(y.shape)
print(x)
print(y)

# %%
model.train()

max_iters = 500

for iter in range(max_iters):
    if iter % 10 == 0 or iter == max_iters - 1:
        print(f"step {iter}")

    _, loss, _ = model(x, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# %%
# generate from the model
context = torch.tensor(encode("paris"), device=device)
context = context.unsqueeze(0)

generated = model.generate(context, max_new_tokens=20)[0].tolist()
decoded = decode(generated)
print(decoded)

# %%
x = torch.tensor(encode("paris is the capital of"), device=device).unsqueeze(0)
logits, loss, attention_weights = model(torch.tensor(x))

# %%
probs_last_token_cities_countries(model=model, cities_countries=cities_countries)

# %% [markdown]
# However, as you can see, the model now predicts that all the cities are the capital of China. This is catastrophic forgetting.

# %%
