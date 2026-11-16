import re
import torch
import torch.nn as nn
import torch.nn.functional as F

with open("saltan.txt", encoding="cp1251") as file:
    text = file.read()

text = "".join(re.findall(r"[а-яё\s]", text))
words = text.split()

vocab = sorted(set(words))
word2idx = {w: i for i, w in enumerate(vocab)}
idx2word = {i: w for w, i in word2idx.items()}
vocab_size = len(vocab)

data = torch.tensor([word2idx[w] for w in words], dtype=torch.long)
print(f"Слов в словаре: {vocab_size}, всего токенов: {len(data)}")


class WordRNN(nn.Module):
    def __init__(self, vocab_size, embed_size=64, hidden_size=128, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.RNN(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        emb = self.embedding(x)
        out, hidden = self.rnn(emb, hidden)
        logits = self.fc(out)
        return logits, hidden


def make_batches(data, seq_len, batch_size):
    n_steps = len(data) // batch_size
    if n_steps <= seq_len:
        return
    data = data[: n_steps * batch_size]
    data = data.view(batch_size, n_steps)
    for i in range(0, n_steps - seq_len):
        x = data[:, i : i + seq_len]
        y = data[:, i + 1 : i + seq_len + 1]
        yield x, y


def train(model, data, epochs=12, seq_len=25, batch_size=32, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    model.train()

    for epoch in range(epochs):
        total_loss = 0.0
        n = 0
        hidden = None
        for x, y in make_batches(data, seq_len, batch_size):
            optimizer.zero_grad()
            logits, hidden = model(x, hidden)
            hidden = hidden.detach()
            loss = criterion(logits.reshape(-1, vocab_size), y.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item()
            n += 1
        print(f"epoch {epoch+1:02d} | loss = {total_loss / max(n,1):.4f}")


@torch.no_grad()
def generate(model, start_words, n_words=50, temperature=0.8):
    model.eval()
    idxs = [word2idx.get(w, 0) for w in start_words.lower().split()]
    x = torch.tensor(idxs, dtype=torch.long).unsqueeze(0)
    logits, hidden = model(x)
    result = start_words.split()

    last = x[:, -1:]
    for _ in range(n_words):
        logits, hidden = model(last, hidden)
        probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)
        next_id = torch.multinomial(probs, 1)
        result.append(idx2word[next_id.item()])
        last = next_id
    return " ".join(result)


if __name__ == "__main__":
    model = WordRNN(vocab_size, embed_size=64, hidden_size=128)
    train(model, data, epochs=12, seq_len=25, batch_size=32, lr=1e-3)

    print("Пример генерации")
    print(generate(model, "дворец", n_words=40))