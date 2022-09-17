import torch

def is_cuda_available():
    if torch.cuda.is_available() is False:
        raise SystemExit("Cuda problem!")
    else:
        print("Cuda is available")
    return True

def check_and_return_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    return device

def get_num_correct(preds, labels):
    return torch.round(preds).argmax(dim=1).eq(labels).sum().item()

def print_tensor_size(tensorname, thetensor):
    print("size of {} = {}".format(tensorname, torch.Tensor.size(thetensor)))