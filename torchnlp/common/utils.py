
def save_attention(name, mod, inp, out):
    attentions[name].append(out.cpu()