import torch
from tqdm.notebook import tqdm


def test_loop(test_loader, model, device):
    y_pred_list = []
    y_true_list = []
    with torch.no_grad():
        for x_batch, y_batch in tqdm(test_loader):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            y_test_pred = model(x_batch)
            y_test_pred = torch.log_softmax(y_test_pred, dim=1)
            _, y_pred_tag = torch.max(y_test_pred, dim=1)

            for i in y_pred_tag.cpu().numpy().tolist():
                y_pred_list.append(i)

            for i in y_batch.cpu().numpy().tolist():
                y_true_list.append(i)

    return y_pred_list, y_true_list

