import torch
import torch.nn as nn
import torch.distributed as dist
from e3nn.o3 import Irreps
import time
import matplotlib.pyplot as plt

class SplitTrainer():

    def __init__(self, backbone, head, head_irreps):

        self.backbone = backbone      # takes atom graph, outputs internal embeddings after message passing
        self.head = head              # takes internal embeddings, outputs fixed irrep size
        self.head_irreps = head_irreps

    def train_backbone(self, 
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
                       num_warmup_epochs=20):
        """
        Train both the backbone and the output head
        """

        print(f"Loss Targets: {node_target_name}, {edge_target_name}" )

        if not val_loader:
            print("Note: using training dataset for scheduler updates")
            val_loader = train_loader
        
        num_train_batches = len(train_loader)
        num_val_batches = len(val_loader)

        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank() 
            self.backbone = nn.parallel.DistributedDataParallel(self.backbone, device_ids=[0], output_device=0, find_unused_parameters=False)
            self.head = nn.parallel.DistributedDataParallel(self.head, device_ids=[0], output_device=0, find_unused_parameters=False)
        else:
            rank = 0

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
            
            self.backbone.train()  
            self.head.train()

            train_loss_node = 0.0
            train_loss_edge = 0.0
            for batch in train_loader:

                optimizer.zero_grad()

                # -- Forward -- 
                batch = batch.to(device)
                backbone_out = self.backbone(batch) 
                node_output, edge_output = self.head(backbone_out["node_embeddings"], backbone_out["edge_embeddings"])

                # -- Loss -- 
                if include_edges:
                    this_node_target = getattr(batch, node_target_name)
                    this_edge_target = getattr(batch, edge_target_name)
                    output = torch.cat([node_output, edge_output], dim=0)
                    labels = torch.cat([this_node_target, this_edge_target], dim=0)
                    loss_node = loss_fxn(node_output, this_node_target)
                    loss_edge = loss_fxn(edge_output, this_edge_target) 
                    loss = loss_fxn(output, labels)

                    train_loss_node += loss_node
                    train_loss_edge += loss_edge

                else:
                    this_node_target = getattr(batch, node_target_name)

                    if self.head_irreps == '1x1e':             # permute force vectors to match edge permutations
                        this_node_target = this_node_target[:, [1, 2, 0]]

                    loss = loss_fxn(node_output, this_node_target)  
                    train_loss_node += loss
                
                # Aggregate loss 
                dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                dist.all_reduce(train_loss_node, op=dist.ReduceOp.SUM)
                loss /= dist.get_world_size()
                train_loss_node /= dist.get_world_size()

                if include_edges:
                    dist.all_reduce(train_loss_edge, op=dist.ReduceOp.SUM)
                    train_loss_edge /= dist.get_world_size()
                
                # -- Backwards -- 
                loss.backward()
                optimizer.step()
                
            # -- Output dump -- 
            if include_edges:
                track_loss_node.append(train_loss_node.cpu().detach().numpy()/num_train_batches) 
                track_loss_edge.append(train_loss_edge.cpu().detach().numpy()/num_train_batches)
            else:
                track_loss_node.append(train_loss_node.cpu().detach().numpy()/num_train_batches) 

            if rank == 0:
                print(f"Epoch {epoch+1}, Train Loss: [node] {track_loss_node[-1]} [edge] {track_loss_edge[-1]}", flush=True)    

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
                    backbone_out = self.backbone(batch) 
                    node_output, edge_output = self.head(backbone_out["node_embeddings"], backbone_out["edge_embeddings"])
                    
                    # -- Loss --
                    if include_edges:
                        this_node_target = getattr(batch, node_target_name)
                        this_edge_target = getattr(batch, edge_target_name)
                        output = torch.cat([node_output, edge_output], dim=0)
                        labels = torch.cat([this_node_target, this_edge_target], dim=0)
                        loss_node = loss_fxn(node_output, this_node_target)
                        loss_edge = loss_fxn(edge_output, this_edge_target) 
                        loss = loss_fxn(output, labels)

                        val_loss_node += loss_node
                        val_loss_edge += loss_edge

                    else:
                        this_node_target = getattr(batch, node_target_name)

                        if self.head_irreps == '1x1e':             # permute force vectors to match edge permutations
                            this_node_target = this_node_target[:, [1, 2, 0]]

                        loss = loss_fxn(node_output, this_node_target)  
                        val_loss_node += loss
                            
                    val_loss += loss.item()
            
            val_loss_tensor = torch.tensor(val_loss, device=device)
            dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
            val_loss = val_loss_tensor.item() / dist.get_world_size()

            # -- Output dump -- 
            if include_edges:
                track_loss_node_val.append(val_loss_node.cpu().detach().numpy()/num_train_batches) 
                track_loss_edge_val.append(val_loss_edge.cpu().detach().numpy()/num_train_batches)
            else:
                track_loss_node_val.append(val_loss_node/num_train_batches) 

            if rank == 0:
                print(f"Epoch {epoch+1}, Val Loss: [node] {track_loss_node_val[-1]} [edge] {track_loss_edge_val[-1]}", flush=True)    

            # -- Scheduler -- 
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            if rank == 0:
                print("Current learning rate: ", current_lr)
            
            epoch_end = time.perf_counter()
            if rank == 0:
                print("Time per epoch: ", epoch_end - epoch_start)
            
            # save state
            if rank == 0:
                if (epoch + 1) % 100 == 0:
                    if include_edges:
                        self.save_training_state(self.backbone, optimizer, track_loss_node, track_loss_node_val, 'backbone', output_folder, track_loss_edge, track_loss_edge_val)
                        self.save_training_state(self.head, optimizer, track_loss_node, track_loss_node_val, 'head', output_folder, track_loss_edge, track_loss_edge_val)     
                    else:
                        self.save_training_state(self.backbone, optimizer, track_loss_node, track_loss_node_val, 'backbone', output_folder)
                        self.save_training_state(self.head, optimizer, track_loss_node, track_loss_node_val, 'head', output_folder)
    
    def adjust_learning_rate(self, optimizer, epoch, warmup_epochs, initial_lr, final_lr):
        """Adjusts the learning rate linearly during the warmup phase."""
        if epoch < warmup_epochs:
            lr = initial_lr + (final_lr - initial_lr) * (epoch / warmup_epochs)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            print(f"Warmup epoch {epoch+1}: setting learning rate to {lr}")

    def save_training_state(self, model, optimizer, track_loss_node, track_validation_node, save_file, output_folder, track_loss_edge=None, track_validation_edge=None):
        """
        Save the training state of the model and optimizer
        """
        torch.save({'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()}, output_folder + "/" + save_file + '.pt')
        torch.save(model.state_dict(), output_folder + "/" + save_file + '_state_dic.pt')


        if track_loss_edge:
            with open(output_folder + "/" + save_file + '_training_loss.txt', 'w') as f:
                for edge, node in zip(track_loss_edge, track_loss_node):
                    f.write(f"{edge:.8f}\t{node:.8f}\n")

            with open(output_folder + "/" + save_file + '_validation_loss.txt', 'w') as f:
                for edge, node in zip(track_validation_edge, track_validation_node):
                    f.write(f"{edge:.8f}\t{node:.8f}\n")
        else:
            with open(output_folder + "/" + save_file + '_training_loss.txt', 'w') as f:
                for node in track_loss_node:
                    f.write(f"{node:.8f}\n")

            with open(output_folder + "/" + save_file + '_validation_loss.txt', 'w') as f:
                for node in track_validation_node:
                    f.write(f"{node:.8f}\n")

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
