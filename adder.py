# %% [markdown]
# # GPT adder

# %% [markdown]
# This repo was inspired by karpathy's suggested exercice in his video [Let's build GPT: from scratch, in code, spelled out.](https://youtu.be/kCc8FmEb1nY?feature=shared)

# %%
import json
import random
from dataclasses import dataclass

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from gpt.model import GPT, GPTConfig, Trainer, TrainingConfig

# %% [markdown]
# ## generate samples
# %% [markdown]
# The following function will generate random samples for training the GPT adder.
#
# It generates the digits of the answer in reverse order, as the typical addition algorithm would proceed right to left too.
#
# i.e. `79+11=09` because `79+11=90`
#
# If `DataConfig.chain_of_thought` is `True`, the answer will be broken down into atomic additions, with the carry in brackets.
#
# i.e. `55+96;5+6=11(1);5+9+1=15;151` because to compute 55+96 you do, in this order:
#
# - 5 + 6 = 11 carry 1
# - 5 + 9 + _carry_ = 5 + 9 + 1 = 15
# - result is 151
# %%


@dataclass
class DataConfig:
    train_low: int = 0
    train_high: int = 1000
    test_low: int = 0
    test_high: int = 1000
    chain_of_thought: bool = True


data_config = DataConfig()


# %%
def generate_sample(low, high, detailed=data_config.chain_of_thought):
    a = random.randint(low, high)
    b = random.randint(low, high)

    if not detailed:
        sample = f"{a}+{b}="
        prompt_end_idx = len(sample) - 1
        res = a + b

        # add in reverse order
        sample += str(res)[::-1]

        return sample, prompt_end_idx

    sample = f"{a}+{b};"

    # give the index of the last character of the prompt (useful for ignore_index of cross entropy loss, see later...)
    prompt_end_idx = len(sample) - 1

    array_of_numbers_a = [int(x) for x in str(a)]
    array_of_numbers_b = [int(x) for x in str(b)]

    carry = 0

    while len(array_of_numbers_a) and len(array_of_numbers_b):
        prefix_carry = f"+{carry}" if carry else ""

        d1 = array_of_numbers_a.pop()
        d2 = array_of_numbers_b.pop()

        res = d1 + d2 + carry

        carry = res // 10

        if len(array_of_numbers_a) == 0 and len(array_of_numbers_b) == 0:
            sample += f"{d1}+{d2}{prefix_carry}={res};"
        else:
            sample += f"{d1}+{d2}{prefix_carry}={res}({carry});"

    while len(array_of_numbers_a):
        d1 = array_of_numbers_a.pop()
        d2 = carry
        res = d1 + d2
        carry = res // 10
        if len(array_of_numbers_a) > 0:
            sample += f"{d1}+{d2}={res}({carry});"
        else:
            sample += f"{d1}+{d2}={res};"

    while len(array_of_numbers_b):
        d1 = array_of_numbers_b.pop()
        d2 = carry
        res = d1 + d2
        carry = res // 10
        if len(array_of_numbers_b) > 0:
            sample += f"{d1}+{d2}={res}({carry});"
        else:
            sample += f"{d1}+{d2}={res};"

    sample += f"{a+b}"

    return sample, prompt_end_idx


# get a sample from the training set
sample, prompt_idx = generate_sample(low=data_config.train_low, high=data_config.train_high)
sample, prompt_idx

# %%
# get a sample from the test set
sample, prompt_idx = generate_sample(low=data_config.test_low, high=data_config.test_high)
sample, prompt_idx


# %%
def extract_prompt_from_sample(sample, prompt_idx):
    """extract only the prompt from a full addition sample"""
    return sample[: prompt_idx + 1]


extract_prompt_from_sample(sample, prompt_idx)


# %%
def correct_answer(sample, model_output, only_result=False):
    """check if the model calculated the correct answer by comparing against the ground truth"""
    if only_result:
        response = f";{sample.split(';')[-1]}<|endoftext|>"
        return response in model_output
    else:
        return model_output == (sample + "<|endoftext|>")


correct_answer(sample, sample + "<|endoftext|>", only_result=True)

# %% [markdown]
# ## build the vocab

# %%
tokens = set()
for i in range(10):
    tokens.add((str(i)))

if data_config.chain_of_thought:
    tokens.add("+")
    tokens.add("=")
    tokens.add(";")
    tokens.add("(")
    tokens.add(")")
else:
    tokens.add("+")
    tokens.add("=")
tokens.add("<|endoftext|>")
tokens

# %% [markdown]
# ## create config

# %%
model_config = GPTConfig(vocab_size=len(tokens), block_size=64, n_embd=128, n_head=4, n_layer=1, dropout=0.2)

train_config = TrainingConfig(
    batch_size=32, max_iters=5000, eval_iters=100, eval_interval=1000, learning_rate=1e-3, weight_decay=1e-1
)

model_config, train_config

# %% [markdown]
# ## tokenizer

# %%
stoi = {s: i for i, s in enumerate(tokens)}
itos = {i: s for i, s in enumerate(tokens)}


def encode(s):
    return [stoi[ch] for ch in s]


def decode(ids):
    return " ".join(itos[i] for i in ids)


# %%
EOS_ID = stoi["<|endoftext|>"]
model_config.eos_token_id = EOS_ID
EOS_ID

# %% [markdown]
# ## model definition

# %%
model = GPT(config=model_config)

# put the model on the device
model.to(model_config.device)

# print the number of parameters in the model
print(sum(p.numel() for p in model.parameters()), "parameters")

# %% [markdown]
# ## data loader and trainer

# %% [markdown]
# The tokens that belongs to the prompt are converted into `-1` in the label vector, as the loss will not be calculated for these tokens (see `ignore_index` of the [cross entropy loss](https://pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html)).


# %%
class AdderTrainer(Trainer):
    def __init__(self, model, model_config, train_config):
        super().__init__(model=model, model_config=model_config, train_config=train_config)

    def get_batch(self, split: str = "train"):
        X, Y = [], []
        for k in range(self.train_config.batch_size):
            sample, prompt_end_idx = generate_sample(low=data_config.train_low, high=data_config.train_high)
            encoded_sample = torch.tensor(encode(sample))

            # add EOS token
            encoded_sample = torch.cat((encoded_sample, torch.tensor([EOS_ID])))

            x = encoded_sample[:-1].clone()
            y = encoded_sample[1:].clone()

            y[:prompt_end_idx] = -1
            X.append(x)
            Y.append(y)

        X = torch.nn.utils.rnn.pad_sequence(X, batch_first=True, padding_value=EOS_ID)
        Y = torch.nn.utils.rnn.pad_sequence(Y, batch_first=True, padding_value=-1)

        X, Y = X.to(self.model_config.device), Y.to(self.model_config.device)
        return X, Y

    @torch.no_grad()
    def estimate_loss(self):
        out = {}
        self.model.eval()
        losses = torch.zeros(self.train_config.eval_iters)
        for k in tqdm(range(self.train_config.eval_iters)):
            X, Y = self.get_batch()
            logits, loss = self.model(X, Y)
            losses[k] = loss.item()
        out["val"] = losses.mean()
        out["train"] = losses.mean()
        self.model.train()
        return out


trainer = AdderTrainer(model, model_config, train_config)

x, y = trainer.get_batch()
x.shape, y.shape

# %%
idx = 2
print(f"y={y[idx]}")
print(f"x={x[idx]}")

without_ignore_index = y[idx].tolist()
without_ignore_index = [i for i in without_ignore_index if i != -1]

print(f"y={decode(without_ignore_index)}")
print(f"x={decode(x[idx].tolist())}")

# %%
train_losses, val_losses = trainer.train()

# %%
plt.plot(val_losses)
plt.xlabel(xlabel="Iteration")
plt.ylabel(ylabel="Loss")
plt.title("Validation Loss")
plt.show()


# %%
def get_model_response(prompt):
    context = torch.tensor(encode(prompt), device=model_config.device)
    context = context.unsqueeze(0)

    generated = model.generate(context, max_new_tokens=70)[0].tolist()
    decoded = decode(generated)
    return decoded


# %%
get_model_response("6+9;")

# %% [markdown]
# ## evaluation


# %%
def accuracy(model: GPT, num_to_run=1000, verbose=False):
    score = 0

    goods, wrongs = [], []

    for _ in tqdm(range(num_to_run)):
        result = {}

        sample, prompt_idx = generate_sample(low=data_config.test_low, high=data_config.test_high)

        prompt = extract_prompt_from_sample(sample, prompt_idx)

        result["prompt"] = prompt

        # generate from the model
        context = torch.tensor(encode(prompt), device=model_config.device)
        context = context.unsqueeze(0)

        generated = model.generate(context, max_new_tokens=70)[0].tolist()
        decoded = decode(generated)

        if verbose:
            print(f"prompt: {prompt}")
            print(f"decoded: {decoded}")
            print(f"sample: {sample}")
            print(f"correct: {correct_answer(sample, decoded)}")
            print()

        result["decoded"] = decoded

        correct = correct_answer(sample, decoded)

        result["correct"] = correct

        if correct:
            score += 1
            goods.append(result)
        else:
            wrongs.append(result)

    return score / num_to_run, goods, wrongs


# %%
score, goods, wrongs = accuracy(model, num_to_run=1000, verbose=False)
print(f"Accuracy: {score*100}%")

# %%
print(len(goods))
print(json.dumps(goods[:10], indent=2))

# %%
print(len(wrongs))
print(json.dumps(wrongs[:10], indent=2))

# %%
