import torch
import torch.nn as nn
import torch.distributed as dist
from torch.cuda.amp import autocast, GradScaler
import time
import matplotlib.pyplot as plt
import numpy as np
from e3nn.o3 import Irreps
import wandb

# note: removing amp to get better precision for now
def disable_amp(func):
    def wrapper(*args, **kwargs):
        print("Disabling torch amp")
        with torch.cuda.amp.autocast(enabled=False):
            return func(*args, **kwargs)
    return wrapper

def get_timestamp_uid() -> str:
    return datetime.datetime.now().strftime("%Y%m-%d%H-%M%S-") + str(uuid4())[:4]

class SplitTrainer():

    def __init__(self, backbone, head, head_irreps, save_frequency=10, run_id=None, run_name='noname'):

        self.backbone = backbone      # takes atom graph, outputs internal embeddings
        self.head = head              # takes internal embeddings, outputs fixed irrep size
        self.head_irreps = head_irreps
        self.save_frequency = save_frequency

        if not run_id:
            run_id = str(get_timestamp_uid)
        
        # config: any dictionary, add the training parameters
        config = {}

        wandb.init(config=config,    
                   id=run_id,
                   name=run_name,
                   project='fockmatrices',
                   entity='manasakani')
        
    @disable_amp
    def train(self, 
            num_epochs, 
            loss_fxn, 
            optimizer, 
            scheduler, 
            device, 
            train_loader, 
            loss_target_string, 
            node_target_name, 
            val_loader=None, 
            edge_target_name=None, 
            output_folder='outputs',
            num_warmup_epochs=0,
            train_backbone=True,
            train_head=True):

        print(f"Loss Targets: {node_target_name}, {edge_target_name}" )

        if not val_loader:
            print("Note: using training dataset for scheduler updates")
            val_loader = train_loader

        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank() 
            if train_backbone:
                self.backbone = nn.parallel.DistributedDataParallel(self.backbone, device_ids=[0], output_device=0, find_unused_parameters=False)
            if train_head:
                self.head = nn.parallel.DistributedDataParallel(self.head, device_ids=[0], output_device=0, find_unused_parameters=False)
        else:
            rank = 0
        
        scaler = GradScaler()  # for mixed precision training

        # Ensure that the ranks have the same number of batches!
        num_train_batches = len(train_loader)
        num_val_batches = len(val_loader)
        self.check_batch_consistency(num_train_batches, num_val_batches, device)

        include_edges = False
        if edge_target_name:
            include_edges = True
        
        warmup_epochs = num_warmup_epochs
        initial_lr = optimizer.param_groups[0]['lr']

        track_loss_node = []
        track_loss_node_val = []
        if include_edges:
            track_loss_edge = []
            track_loss_edge_val = []
        
        for epoch in range(num_epochs):
            epoch_start = time.perf_counter()

            self.adjust_learning_rate(optimizer, epoch, warmup_epochs, initial_lr, initial_lr*10)
            
            if train_backbone:
                self.backbone.train() 
                
            if train_head: 
                self.head.train()

            train_loss_node = 0.0
            train_loss_edge = 0.0
            for batch in train_loader:

                optimizer.zero_grad()

                # -- Forward -- 
                batch = batch.to(device)
                with autocast():
                    backbone_out = self.backbone(batch) 

                    if include_edges:
                        node_output, edge_output = self.head(backbone_out, batch)

                        this_node_target = getattr(batch, node_target_name)
                        edge_mask = batch.edge_mask
                        this_edge_target = getattr(batch, edge_target_name)[edge_mask]

                        output = torch.cat([node_output, edge_output], dim=0)
                        labels = torch.cat([this_node_target, this_edge_target], dim=0)
                        loss_node = loss_fxn(node_output, this_node_target)
                        loss_edge = loss_fxn(edge_output, this_edge_target) 
                        loss = loss_fxn(output, labels)

                        train_loss_node += loss_node
                        train_loss_edge += loss_edge

                    else:
                        node_output = self.head(backbone_out, batch)
                        this_node_target = getattr(batch, node_target_name)

                        if self.head_irreps == '1x1e':             
                            this_node_target = this_node_target[:, [1, 2, 0]] # match edge permutations
                            loss = loss_fxn(node_output['forces'], this_node_target) 
                        else:
                            print("To be implemented!") 

                        train_loss_node += loss
                    
                # -- Backwards -- 
                # loss.backward()
                # optimizer.step()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
            # -- Output dump -- 
            if include_edges:
                track_loss_node.append(train_loss_node.cpu().detach().numpy()/num_train_batches) 
                track_loss_edge.append(train_loss_edge.cpu().detach().numpy()/num_train_batches)
            else:
                track_loss_node.append(train_loss_node.cpu().detach().numpy()/num_train_batches) 

            if rank == 0:
                if include_edges:
                    print(f"Epoch {epoch+1}, Train Loss: [node] {track_loss_node[-1]} [edge] {track_loss_edge[-1]}", flush=True)    
                else:
                    print(f"Epoch {epoch+1}, Train Loss: [node] {track_loss_node[-1]}", flush=True)    

            dist.barrier()

            # Validation step
            self.backbone.eval()
            self.head.eval()
            val_loss = 0.0
            val_loss_node = 0.0
            val_loss_edge = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    
                    # -- Forward --
                    batch = batch.to(device)
                    with autocast():
                        backbone_out = self.backbone(batch) 
                        
                        # -- Loss --
                        if include_edges:
                            node_output, edge_output = self.head(backbone_out, batch)
                            this_node_target = getattr(batch, node_target_name)
                            edge_mask = batch.edge_mask
                            this_edge_target = getattr(batch, edge_target_name)[edge_mask]

                            output = torch.cat([node_output, edge_output], dim=0)
                            labels = torch.cat([this_node_target, this_edge_target], dim=0)
                            loss_node = loss_fxn(node_output, this_node_target)
                            loss_edge = loss_fxn(edge_output, this_edge_target) 
                            loss = loss_fxn(output, labels)

                            val_loss_node += loss_node
                            val_loss_edge += loss_edge

                        else:
                            node_output = self.head(backbone_out, batch)
                            this_node_target = getattr(batch, node_target_name)

                            if self.head_irreps == '1x1e':             # permute force vectors to match edge permutations
                                this_node_target = this_node_target[:, [1, 2, 0]]
                                loss = loss_fxn(node_output['forces'], this_node_target) 
                            else:
                                print("To be implemented!")  

                            val_loss_node += loss
                                
                        val_loss += loss.item()
            
            # -- Output dump -- 
            if include_edges:
                track_loss_node_val.append(val_loss_node.cpu().detach().numpy()/num_val_batches) 
                track_loss_edge_val.append(val_loss_edge.cpu().detach().numpy()/num_val_batches)
            else:
                track_loss_node_val.append(val_loss_node.cpu().detach().numpy()/num_val_batches) 

            if rank == 0:
                if include_edges:
                    print(f"Epoch {epoch+1}, Val Loss: [node] {track_loss_node_val[-1]} [edge] {track_loss_edge_val[-1]}", flush=True)    
                else:
                    print(f"Epoch {epoch+1}, Val Loss: [node] {track_loss_node_val[-1]}", flush=True)    

            # -- Scheduler -- 
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            if rank == 0:
                print("Current learning rate: ", current_lr)
            
            epoch_end = time.perf_counter()
            if rank == 0:
                print("Time per epoch: ", epoch_end - epoch_start)

            
            # log:
            update_dict = {"node_loss": float(track_loss_node[-1]), 
                            "node_val_loss": float(track_loss_node_val[-1]),
                            "edge_loss": float(track_loss_edge[-1]), 
                            "edge_val_loss": float(track_loss_edge_val[-1])}

            # add some more stuff to the dictionary
            wandb.log(update_dict)
            
            # save state
            if rank == 0:
                if (epoch + 1) % self.save_frequency == 0:
                    if include_edges:
                        self.save_training_state(epoch, self.backbone, optimizer, track_loss_node, track_loss_node_val, 'backbone', output_folder, track_loss_edge, track_loss_edge_val)
                        self.save_training_state(epoch, self.head, optimizer, track_loss_node, track_loss_node_val, 'head', output_folder, track_loss_edge, track_loss_edge_val)     
                    else:
                        self.save_training_state(epoch, self.backbone, optimizer, track_loss_node, track_loss_node_val, 'backbone', output_folder)
                        self.save_training_state(epoch, self.head, optimizer, track_loss_node, track_loss_node_val, 'head', output_folder)
    
    def check_batch_consistency(self, num_train_batches, num_val_batches, device):

        if dist.is_available() and dist.is_initialized():
            train_batches_tensor = torch.tensor([num_train_batches], device=device)
            val_batches_tensor = torch.tensor([num_val_batches], device=device)
            train_batches_list = [torch.zeros_like(train_batches_tensor) for _ in range(dist.get_world_size())]
            val_batches_list = [torch.zeros_like(val_batches_tensor) for _ in range(dist.get_world_size())]
            dist.all_gather(train_batches_list, train_batches_tensor)
            dist.all_gather(val_batches_list, val_batches_tensor)

            dist.barrier()

            if not all(train_batches_list[0] == tb for tb in train_batches_list):
                print("Mismatch in number of training batches across ranks!", flush=True)
                raise ValueError("Mismatch in number of training batches across ranks!", flush=True)
            if not all(val_batches_list[0] == vb for vb in val_batches_list):
                print("Mismatch in number of validation batches across ranks!", flush=True)
                raise ValueError("Mismatch in number of validation batches across ranks!", flush=True)

    def adjust_learning_rate(self, optimizer, epoch, warmup_epochs, initial_lr, final_lr):
        """Adjusts the learning rate linearly during the warmup phase."""
        if epoch < warmup_epochs:
            lr = initial_lr + (final_lr - initial_lr) * (epoch / warmup_epochs)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            print(f"Warmup epoch {epoch+1}: setting learning rate to {lr}")
    
    def evaluate(self, 
                loss_fxn, 
                device, 
                eval_loader, 
                loss_target_string, 
                node_target_name, 
                edge_target_name=None, 
                basis_transform=None,
                output_folder='outputs'):
        
        print(f"Loss Targets: {node_target_name}, {edge_target_name}" )
        print("Running eval.")
        self.backbone.eval() 
        self.head.eval() 

        num_eval_batches = len(eval_loader)

        include_edges = False
        if edge_target_name:
            include_edges = True

        track_loss = []
        track_loss_node = []
        if include_edges:
            track_loss_edge = []
        
        # -- Evaluate everything in the train_loader -- 
        with torch.no_grad():  
            train_loss_node = 0.0
            train_loss_edge = 0.0
            train_loss = 0.0

            for index, batch in enumerate(eval_loader):

                batch = batch.to(device)
                backbone_out = self.backbone(batch) 

                # check
                self.visualize_embeddings(backbone_out["node_embeddings"][0:3], output_folder, keyword='node')
                self.visualize_embeddings(backbone_out["edge_embeddings"][0:5], output_folder, keyword='edge')

                if include_edges:
                    node_output, edge_output = self.head(backbone_out, batch)

                    # plt.imshow(np.log(np.abs(edge_output[0].detach().reshape(14, 14).cpu().numpy())))
                    # plt.savefig("edge_01.png", dpi=300, bbox_inches='tight')
                    # plt.close()
                    # plt.imshow(np.log(np.abs(edge_output[3].detach().reshape(14, 14).cpu().numpy())))
                    # plt.savefig("edge_10.png", dpi=300, bbox_inches='tight')
                    # plt.close()
                    # print("printed edges")
                    # exit()

                    this_node_target = getattr(batch, node_target_name)
                    this_edge_target = getattr(batch, edge_target_name)

                    # Transform back to uncoupled basis:
                    uncoupled_node_outputs = basis_transform.get_H(node_output)
                    uncoupled_edge_outputs = basis_transform.get_H(edge_output)
                    uncoupled_node_labels = basis_transform.get_H(this_node_target)
                    uncoupled_edge_labels = basis_transform.get_H(this_edge_target)

                    output = torch.cat([uncoupled_node_outputs, uncoupled_edge_outputs], dim=0)
                    labels = torch.cat([uncoupled_node_labels, uncoupled_edge_labels], dim=0)
                    loss_node = loss_fxn(uncoupled_node_outputs, uncoupled_node_labels)
                    loss_edge = loss_fxn(uncoupled_edge_outputs, uncoupled_edge_labels) 
                    loss = loss_fxn(output, labels)

                    train_loss_node += loss_node
                    train_loss_edge += loss_edge
                    train_loss += loss

                else:
                    node_output = self.head(backbone_out, batch)
                    this_node_target = getattr(batch, node_target_name)

                    if self.head_irreps == '1x1e':             
                        this_node_target = this_node_target[:, [1, 2, 0]] # match edge permutations
                        loss = loss_fxn(node_output['forces'], this_node_target) 
                    else:
                        print("To be implemented!") 

                    train_loss += loss
                
                # -- Track -- 
                if include_edges:
                    track_loss_node.append(train_loss_node.cpu().detach().numpy()/num_eval_batches) 
                    track_loss_edge.append(train_loss_edge.cpu().detach().numpy()/num_eval_batches)
                    track_loss.append(train_loss.cpu().detach().numpy()/num_eval_batches) 
                else:
                    track_loss.append(train_loss.cpu().detach().numpy()/num_eval_batches) 

                # remove from gpu
                del batch, node_output
                if include_edges:
                    del edge_output
                torch.cuda.empty_cache()

        # -- Output dump -- 
        if include_edges:
            with open(output_folder + "/" + 'model' + '_eval.txt', 'w') as f:
                    for edge, node, total in zip(track_loss_edge, track_loss_node, track_loss):
                        f.write(f"{edge:.10f}\t{node:.10f}\t{total:.10f}\n")
        else:
            with open(output_folder + "/" + 'model' + '_eval.txt', 'w') as f:
                    for node in track_loss:
                        f.write(f"{node:.10f}\n")
                
    def visualize_embeddings(self, embs, output_folder, keyword):

        for i, emb in enumerate(embs):
            plt.imshow(emb.cpu().detach().numpy(), cmap='RdBu', vmin=-1.0, vmax=1.0)
            plt.savefig(output_folder+"/" + keyword + "_emb_"+str(i)+".png", dpi=300, bbox_inches='tight')
            plt.close()

    def save_training_state(self, step, model, optimizer, track_loss_node, track_validation_node, save_file, output_folder, track_loss_edge=None, track_validation_edge=None):
        """
        Save the training state of the model and optimizer
        """
        torch.save({'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()}, output_folder + "/" + save_file + '.pt')
        torch.save(model.state_dict(), output_folder + "/" + save_file + '_state_dic.pt')

        if track_loss_edge:
            with open(output_folder + "/" + save_file + '_training_loss.txt', 'w') as f:
                for edge, node in zip(track_loss_edge, track_loss_node):
                    f.write(f"{edge:.10f}\t{node:.10f}\n")

            with open(output_folder + "/" + save_file + '_validation_loss.txt', 'w') as f:
                for edge, node in zip(track_validation_edge, track_validation_node):
                    f.write(f"{edge:.10f}\t{node:.10f}\n")
        else:
            with open(output_folder + "/" + save_file + '_training_loss.txt', 'w') as f:
                for node in track_loss_node:
                    f.write(f"{node:.10f}\n")

            with open(output_folder + "/" + save_file + '_validation_loss.txt', 'w') as f:
                for node in track_validation_node:
                    f.write(f"{node:.10f}\n")

        plt.figure(figsize=(4, 3))
        plt.plot(track_loss_node, '-', c='blue', label='node')

        if track_loss_edge:
            plt.plot(track_loss_edge, '-', c='red', label='edge')

        plt.plot(track_validation_node, '--', c='blue', label='validation node')
        if track_loss_edge:
            plt.plot(track_validation_edge,  '--', c='red', label='validation edge')
            
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.yscale('log')
        plt.legend(frameon=False)
        plt.savefig(output_folder + "/" + save_file + '_loss.png', dpi=300, bbox_inches='tight')
        plt.close()
