# Usage (on Kaggle or anywhere with PyTorch):
#   python tools/inspect_checkpoints.py brain_tumor_model_fold_1.pth brain_tumor_model_fold_2.pth ... brain_tumor_model_fold_5.pth
# Prints, per fold checkpoint (state_dict only, loaded safely with weights_only=True):
#   - BatchNorm num_batches_tracked  -> number of training batches seen before the saved (best) checkpoint
#     -> divided by batches per epoch gives the epoch at which the best checkpoint was taken
#   - learned GNN path weights softmax(alpha), softmax(beta)  (order as in g_enc: GCN, GAT, TransformerConv)
#   - uncert.log_tau and the temperature it implies under exp() and softplus()
#   - parameter count per module (buffers excluded)
import sys, math, torch

BATCHES_PER_EPOCH = math.ceil((12064 - 2413) * 0.92 / 36)   # ~8,879 training images per fold, batch size 36 -> 247

def softmax(t):
    return torch.softmax(t.float(), 0).tolist()

for path in sys.argv[1:]:
    sd = torch.load(path, map_location="cpu", weights_only=True)
    steps = sorted({int(v) for k, v in sd.items() if k.endswith("num_batches_tracked")})
    params = {}
    for k, v in sd.items():
        if k.endswith(("running_mean", "running_var", "num_batches_tracked")):
            continue
        params[k.split(".")[0]] = params.get(k.split(".")[0], 0) + v.numel()
    lt = float(sd["uncert.log_tau"])
    print(f"== {path}")
    print(f"   num_batches_tracked values: {steps}  -> best checkpoint after ~{steps[-1] / BATCHES_PER_EPOCH:.1f} epochs "
          f"(assuming {BATCHES_PER_EPOCH} batches/epoch)")
    print(f"   softmax(alpha) = {[round(x, 4) for x in softmax(sd['g_enc.alpha'])]}   "
          f"softmax(beta) = {[round(x, 4) for x in softmax(sd['g_enc.beta'])]}")
    print(f"   log_tau = {lt:.4f}  -> tau = exp: {math.exp(lt):.4f} | softplus: {math.log1p(math.exp(lt)):.4f}")
    print(f"   parameters per module: { {k: f'{v:,}' for k, v in params.items()} }  total {sum(params.values()):,}")
