import torch
import random
from model_definition import CGCNN
from utils import get_loader, get_uncertainty_scores


# def train_active_learning(dataset, initial_labeled_ratio=0.1, query_size=10, cycles=5, epochs=20):
#     total_indices = list(range(len(dataset)))
#     labeled_indices = random.sample(total_indices, int(initial_labeled_ratio * len(dataset)))
#     unlabeled_indices = list(set(total_indices) - set(labeled_indices))

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     model = CGCNN().to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
#     criterion = torch.nn.MSELoss()

#     for cycle in range(cycles):
#         print(f"\n[Cycle {cycle + 1}/{cycles}] Labeled: {len(labeled_indices)} | Unlabeled: {len(unlabeled_indices)}")

#         train_loader = get_loader(dataset, labeled_indices)

#         for epoch in range(1, epochs + 1):
#             model.train()
#             total_loss = 0
#             for batch in train_loader:
#                 batch = batch.to(device)
#                 optimizer.zero_grad()
#                 out = model(batch)
#                 loss = criterion(out.view(-1), batch.y.view(-1))
#                 loss.backward()
#                 optimizer.step()
#                 total_loss += loss.item() * batch.num_graphs
#             print(f"Epoch {epoch:03d} | Train Loss: {total_loss / len(train_loader.dataset):.4f}")

#         if len(unlabeled_indices) == 0:
#             break

#         scores = get_uncertainty_scores(model, dataset, unlabeled_indices, device)
#         selected = [idx for idx, _ in scores[:query_size]]
#         print(f"Queried samples: {selected}")

#         labeled_indices += selected
#         unlabeled_indices = list(set(unlabeled_indices) - set(selected))

#     torch.save(model.state_dict(), "cgcnn_final.pt")
#     print("Active learning complete.")

def train_active_learning(dataset, initial_labeled_ratio=0.1, query_size=10, cycles=5, epochs=20):
    total_indices = list(range(len(dataset)))
    labeled_indices = random.sample(total_indices, int(initial_labeled_ratio * len(dataset)))
    unlabeled_indices = list(set(total_indices) - set(labeled_indices))

    # Train/val split from labeled pool
    train_labeled, val_labeled = train_test_split(labeled_indices, test_size=0.1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CGCNN(output_dim=OUTPUT_DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()

    all_train_losses = []
    all_val_losses = []

    for cycle in range(cycles):
        print(f"\n[Cycle {cycle + 1}/{cycles}] Labeled: {len(labeled_indices)} | Unlabeled: {len(unlabeled_indices)}")

        train_loader = get_loader(dataset, train_labeled)
        val_loader = get_loader(dataset, val_labeled)

        
        for epoch in range(1, epochs + 1):
            model.train()
            total_loss = 0
            for batch in train_loader:
                batch = batch.to(device)
                out = model(batch)
                print(f"out.shape: {out.shape}, batch.y.shape: {batch.y.shape}")  # Debug print
                optimizer.zero_grad()
                # loss = criterion(out, batch.y)
                loss = criterion(out, batch.y.view(out.shape))
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * batch.num_graphs

            train_loss = total_loss / len(train_loader.dataset)
            val_loss = evaluate_loss(model, val_loader, criterion, device)
            all_train_losses.append(train_loss)
            all_val_losses.append(val_loss)
            print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if len(unlabeled_indices) == 0:
            break

        scores = get_uncertainty_scores(model, dataset, unlabeled_indices, device)
        selected = [idx for idx, _ in scores[:query_size]]
        print(f"Queried samples: {selected}")

        labeled_indices += selected
        unlabeled_indices = list(set(unlabeled_indices) - set(selected))
        train_labeled, val_labeled = train_test_split(labeled_indices, test_size=0.1)

    print("Active learning complete.")