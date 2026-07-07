import torch
import torch.nn.functional as F

def compute_flm_loss(model, input_ids, vocab_size, time_warper=lambda t: t):
    B, L = input_ids.shape
    input_ids = input_ids.to(device)

    x1 = F.one_hot(input_ids, num_classes=vocab_size).float()  # (B, L, vocab_size)

    x0 = torch.randn_like(x1).to(device)  # (B, L, vocab_size)

    t = torch.rand(B, 1, 1, device=device)  # (B, 1, 1)
    t_warp = time_warper(t)  # (B, 1, 1)

    It = t_warp * x1 + (1 - t_warp) * x0  # (B, L, vocab_size)

    logits = model(It, t_warp.squeeze(-1).squeeze(-1))

    loss = F.cross_entropy(logits.view(-1, vocab_size), input_ids.view(-1))
    return loss


@torch.no_grad()
def sample_flm_text(model, vocab, num_samples=5, seq_len=4, num_steps=50, time_warper=lambda t: t):
    model.eval()
    vocab_size = len(vocab)
    
    xt = torch.randn(num_samples, seq_len, vocab_size, device=device)  # Start from noise

    t_seq = torch.linspace(0, 1.0, num_steps + 1, device=device)

    for i in range(num_steps):
        t_curr = t_seq[i].item()
        t_next = t_seq[i + 1].item()

        t_curr_warp = time_warper(torch.tensor([t_curr], device=device)).item()
        t_next_warp = time_warper(torch.tensor([t_next], device=device)).item()

        if t_curr >= 1.0:
            break

        t_tensor = torch.full((num_samples,), t_curr, device=device)
        t_warp = time_warper(t_tensor)

        logits = model(xt, t_warp)
        x1_hat = F.softmax(logits, dim=-1)

        bt = (x1_hat - xt) / (1.0 - t_curr_warp)

        xt = xt + bt * (t_next_warp - t_curr_warp)
    
    sample_ids = torch.argmax(xt, dim=-1)  # (num_samples, seq_len)
    sample_sentences = []
    for ids in sample_ids:
        words = [vocab[i] for i in ids.cpu().numpy()]
        sample_sentences.append(' '.join(words))
        
    return sample_sentences