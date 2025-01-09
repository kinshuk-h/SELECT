import torch

def find_parameter(model, name):
    for n, p in model.named_parameters():
        if n == name: return p
    raise LookupError(name)

def add_delta(model, deltas):
    weights_copy = {}
    with torch.no_grad():
        for w_name, upd_matrix in deltas.items():
            w = find_parameter(model, w_name)
            weights_copy[w_name] = w.detach().clone().cpu()
            w[...] += upd_matrix.to(model.device)
    return weights_copy

def restore_weights(model, weights):
    with torch.no_grad():
        for w_name, upd_matrix in weights.items():
            w = find_parameter(model, w_name)
            w[...] = upd_matrix.to(model.device)
