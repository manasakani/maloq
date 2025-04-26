import torch
import matplotlib.pyplot as plt

def save_training_state(model, optimizer, track_loss_node, track_validation_loss_node, save_file):
    """
    Save the training state of the model and optimizer
    """
    torch.save({'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()}, save_file + '.pt')
    torch.save(model.state_dict(), save_file + '_state_dic.pt')

    # with open(save_file + '_training_loss.txt', 'w') as f:
    #     for edge, node in zip(track_loss_edge, track_loss_node):
    #         f.write(f"{edge:.8f}\t{node:.8f}\n")

    # with open(save_file + '_validation_loss.txt', 'w') as f:
    #     for edge, node in zip(track_validation_edge, track_validation_node):
    #         f.write(f"{edge:.8f}\t{node:.8f}\n")

    with open(save_file + '_training_loss.txt', 'w') as f:
        for node in track_loss_node:
            f.write(f"{node:.8f}\n")

    with open(save_file + '_validation_loss.txt', 'w') as f:
        for node in track_validation_loss_node:
            f.write(f"{node:.8f}\n")

    plt.figure(figsize=(4, 3))
    plt.plot(track_loss_node, label='node')
    # plt.plot(track_loss_edge, label='edge')
    plt.plot(track_validation_loss_node, label='validation node')
    # plt.plot(track_validation_edge, label='validation edge')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend()
    plt.savefig(save_file + '_loss.png', dpi=300, bbox_inches='tight')
    plt.close()