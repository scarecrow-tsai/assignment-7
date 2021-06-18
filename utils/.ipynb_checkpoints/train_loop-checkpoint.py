import torch
from tqdm.notebook import tqdm


def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.softmax(y_pred, dim=1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim=1)

    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)

    acc = torch.round(acc * 100)

    return acc


################################
##  TRAIN LOOP
################################
def train_loop(
    model, epochs, optimizer, criterion, scheduler, train_loader, val_loader, device
):

    acc_stats = {"train": [], "val": []}
    loss_stats = {"train": [], "val": []}

    print("\nBegin training.")

    for e in tqdm(range(1, epochs + 1)):

        # TRAINING
        train_epoch_loss = 0
        train_epoch_acc = 0

        model.train()
        for X_train_batch, y_train_batch in train_loader:
            X_train_batch, y_train_batch = (
                X_train_batch.to(device),
                y_train_batch.to(device),
            )

            optimizer.zero_grad()

            y_train_pred = model(X_train_batch).squeeze()

            train_loss = criterion(y_train_pred, y_train_batch)
            train_acc = multi_acc(y_train_pred, y_train_batch)

            train_loss.backward()
            optimizer.step()

            train_epoch_loss += train_loss.item()
            train_epoch_acc += train_acc.item()

        # VALIDATION
        with torch.no_grad():
            model.eval()
            val_epoch_loss = 0
            val_epoch_acc = 0
            for X_val_batch, y_val_batch in val_loader:
                X_val_batch, y_val_batch = (
                    X_val_batch.to(device),
                    y_val_batch.to(device),
                )

                y_val_pred = model(X_val_batch).squeeze()

                val_loss = criterion(y_val_pred, y_val_batch)
                val_acc = multi_acc(y_val_pred, y_val_batch)

                val_epoch_loss += val_loss.item()
                val_epoch_acc += val_acc.item()

        scheduler.step()

        avg_train_epoch_loss = train_epoch_loss / len(train_loader)
        avg_val_epoch_loss = val_epoch_loss / len(val_loader)
        avg_train_epoch_acc = train_epoch_acc / len(train_loader)
        avg_val_epoch_acc = val_epoch_acc / len(val_loader)

        loss_stats["train"].append(avg_train_epoch_loss)
        loss_stats["val"].append(avg_val_epoch_loss)
        acc_stats["train"].append(avg_train_epoch_acc)
        acc_stats["val"].append(avg_val_epoch_acc)

        print(
            f"Epoch {e+0:02}/{epochs}: | Train Loss: {avg_train_epoch_loss:.5f} | Val Loss: {avg_val_epoch_loss:.5f} | Train Acc: {avg_train_epoch_acc:.3f}% | Val Acc: {avg_val_epoch_acc:.3f}%"
        )

    return model, loss_stats, acc_stats
