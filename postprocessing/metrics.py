import matplotlib.pyplot as plt
import pandas as pd

def plot_metric(
    metric,
    aspects_list
    ):
    
    plt.figure(figsize = (10, 4))
    for apsect in aspects_list:
        data = pd.read_csv("./postprocessing/train_history/"+metric+"/"+metric+"_history_"+apsect+".csv")
        plt.plot(data["train"], label = "Train")
        plt.plot(data["val"], label = "Val")
    plt.title("Training Losses for # aspects")
    plt.xlabel("Epochs")
    plt.ylabel("Average Loss")
    plt.legend()
    plt.savefig("./figures/"+metric+"_plot.pdf", bbox_inches="tight")
    plt.show()