import argparse
import os
import time
import torch
import torch.nn.functional as F
from tqdm import tqdm
import threading
from queue import Queue
from sage import SAGE

from lib.data import *
from lib.neighbor_sampler import MMAPNeighborSampler
from lib.utils import *


# Parse arguments
argparser = argparse.ArgumentParser()
argparser.add_argument('--gpu', type=int, default=0)
argparser.add_argument('--num-epochs', type=int, default=10)
argparser.add_argument('--batch-size', type=int, default=1000)
argparser.add_argument('--num-workers', type=int, default=32)
argparser.add_argument('--num-hiddens', type=int, default=256)
argparser.add_argument('--dataset', type=str, default='ogbn-papers100M')
argparser.add_argument('--sizes', type=str, default='10,10,10')
argparser.add_argument('--ginex-num-threads', type=int, default=128)
argparser.add_argument('--train-only', dest='train_only', default=False, action='store_true')
args = argparser.parse_args()

# Set environment and path
os.environ['GINEX_NUM_THREADS'] = str(args.ginex_num_threads)
dataset_path = os.path.join('./dataset', args.dataset + '-ginex')
split_idx_path = os.path.join(dataset_path, 'split_idx.pth')

# Prepare dataset
indptr, indices, x, y, num_features, num_classes, num_nodes, train_idx, valid_idx, test_idx = get_mmap_dataset(path=dataset_path, split_idx_path=split_idx_path)
sizes = [int(size) for size in args.sizes.split(',')]
train_loader = MMAPNeighborSampler(indptr, indices, node_idx=train_idx,
                               sizes=sizes, batch_size=args.batch_size,
                               shuffle=True, num_workers=args.num_workers)
test_loader = MMAPNeighborSampler(indptr, indices, node_idx=test_idx,
                               sizes=sizes, batch_size=args.batch_size,
                               shuffle=False, num_workers=args.num_workers)
valid_loader = MMAPNeighborSampler(indptr, indices, node_idx=valid_idx,
                               sizes=sizes, batch_size=args.batch_size,
                               shuffle=False, num_workers=args.num_workers)

# Define model
device = torch.device('cuda:%d' % args.gpu)
torch.cuda.set_device(device)
model = SAGE(num_features, args.num_hiddens, num_classes, num_layers=len(sizes))
model = model.to(device)


def gather(gather_q, ids, batch_size):
    batch_inputs = gather_mmap(x, ids)
    batch_labels = y[ids[:batch_size]]
    gather_q.put((batch_inputs, batch_labels))


def train(epoch):
    model.train()

    pbar = tqdm(total=train_idx.size(0))
    pbar.set_description(f'Epoch {epoch:02d}')

    total_loss = total_correct = 0

    # Queues for parallel execution of CPU & GPU operations via multi-threading
    adjs_q = Queue(maxsize=2)
    gather_q = Queue(maxsize=1)

    # Sample
    for step, (batch_size, ids, adjs) in enumerate(train_loader):
        if step == 0:
            adjs_q.put(adjs)

            # Gather
            gather_loader = threading.Thread(target=gather, args=(gather_q, ids, batch_size), daemon=True)
            gather_loader.start()

            continue

        adjs_q.put(adjs)

        # Gather
        # Fetch gather results the queue & launch new thread for gather for the next iteration
        (batch_inputs, batch_labels) = gather_q.get()
        gather_loader = threading.Thread(target=gather, args=(gather_q, ids, batch_size), daemon=True)
        gather_loader.start()

        # Transfer
        batch_inputs_cuda = batch_inputs.to(device)
        batch_labels_cuda = batch_labels.to(device)
        adjs = adjs_q.get()
        adjs_cuda = [adj.to(device) for adj in adjs]

        # Forward
        out = model(batch_inputs_cuda, adjs_cuda)
        loss = F.nll_loss(out, batch_labels_cuda.long())

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Free
        total_loss += float(loss)
        total_correct += int(out.argmax(dim=-1).eq(batch_labels_cuda.long()).sum())
        tensor_free(batch_inputs)
        del(batch_inputs_cuda)
        del(ids)
        del(adjs)
        del(batch_labels_cuda)
        torch.cuda.empty_cache()
        pbar.update(batch_size)

    pbar.close()

    loss = total_loss / len(train_loader)
    approx_acc = total_correct / train_idx.size(0)

    return loss, approx_acc


@torch.no_grad()
def inference(mode='test'):
    model.eval()

    if mode == 'test':
        pbar = tqdm(total=test_idx.size(0))
    elif mode == 'valid':
        pbar = tqdm(total=valid_idx.size(0))
    pbar.set_description('Evaluating')

    total_loss = total_correct = 0

    if mode == 'test':
        inference_loader = test_loader
    elif mode == 'valid':
        inference_loader = valid_loader

    # Sample
    for step, (batch_size, ids, adjs) in enumerate(inference_loader):
        # Gather
        batch_inputs = gather_mmap(x, ids)
        batch_labels = y[ids[:batch_size]]

        # Transfer
        batch_inputs_cuda = batch_inputs.to(device)
        batch_labels_cuda = batch_labels.to(device)
        adjs = [adj.to(device) for adj in adjs]

        # Forward
        out = model(batch_inputs_cuda, adjs)
        loss = F.nll_loss(out, batch_labels_cuda.long())
        tensor_free(batch_inputs)

        torch.cuda.synchronize()
        total_loss += float(loss)
        total_correct += int(out.argmax(dim=-1).eq(batch_labels_cuda.long()).sum())
        pbar.update(batch_size)

    pbar.close()

    loss = total_loss / len(inference_loader)
    if mode == 'test':
        approx_acc = total_correct / test_idx.size(0)
    elif mode == 'valid':
        approx_acc = total_correct / valid_idx.size(0)

    return loss, approx_acc


if __name__=='__main__':
    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

    best_val_acc = final_test_acc = 0
    for epoch in range(args.num_epochs):
        start = time.time()
        loss, acc = train(epoch)
        end = time.time()
        print(f'Epoch {epoch:02d}, Loss: {loss:.4f}, Approx. Train: {acc:.4f}')
        print('Epoch time: {:.4f} ms'.format((end - start) * 1000))

        if epoch > 3 and not args.train_only:
            val_loss, val_acc = inference(mode='valid')
            test_loss, test_acc = inference(mode='test')
            print ('Valid loss: {0:.4f}, Valid acc: {1:.4f}, Test loss: {2:.4f}, Test acc: {3:.4f},'.format(val_loss, val_acc, test_loss, test_acc))

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                final_test_acc = test_acc

    if not args.train_only:
        print('Final Test acc: {final_test_acc:.4f}')
