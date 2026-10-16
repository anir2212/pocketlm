import torch

from scratch.tiny_gpt import GPTConfig, TinyGPT, decode, encode


def test_future_tokens_cannot_change_past_logits():
    model = TinyGPT(GPTConfig(vocab_size=257, block_size=8, n_layer=2,
                              n_head=2, n_embd=32, dropout=0.0)).eval()
    before = torch.tensor([[1, 2, 3, 4, 5]])
    after = torch.tensor([[1, 2, 3, 99, 100]])
    with torch.no_grad():
        a, _ = model(before)
        b, _ = model(after)
    assert torch.allclose(a[:, :3], b[:, :3], atol=1e-6)


def test_byte_tokenizer_round_trip():
    text = "Once upon a time 🌙"
    assert decode(encode(text)) == text


def test_configured_model_is_between_ten_and_twenty_million_parameters():
    model = TinyGPT(GPTConfig())
    count = sum(parameter.numel() for parameter in model.parameters())
    assert 10_000_000 <= count <= 20_000_000
