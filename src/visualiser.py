import matplotlib.pyplot as plt
import torch
from torchviz import make_dot
import os

class Visualizer:
    '''
    Visualisation methods for:
    Plotting model history (timeline).
    Visualising the model connections.    
    '''
    @staticmethod
    def plot_history(model_history):
        accuracy = [entry[0] for entry in model_history]
        loss = [entry[1] for entry in model_history]

        fig, ax1 = plt.subplots(figsize=(8, 5))

        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy', color='tab:blue')
        ax1.plot(accuracy, label='Accuracy', color='tab:blue')
        ax1.tick_params(axis='y', labelcolor='tab:blue')

        ax2 = ax1.twinx()
        ax2.set_ylabel('Loss', color='tab:red')
        ax2.plot(loss, label='Loss', color='tab:red')
        ax2.tick_params(axis='y', labelcolor='tab:red')

        plt.title('Model Accuracy and Loss')
        fig.tight_layout()
        plt.show()

    @staticmethod
    def visualize_model(model, input_size, vocab_size, device):
        x = torch.randint(0, vocab_size, input_size).to(device)
        y = model(x)

        # Install Graphviz. If you have it then you have issues with the PATH variables, just set it by hand
        # os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'
        os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'
        make_dot(y, params=dict(model.named_parameters())).render("model", format="png")
