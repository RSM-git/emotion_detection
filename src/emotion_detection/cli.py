import os

import questionary

def start_up():
    answer = questionary.select("What do you want to do?",
                                choices=["Train model", "Inference"]
                                ).ask()
    return answer


def select_model():
    model_name = questionary.select("Which model?",
                                    choices=os.listdir("models")+"Go back",
                                    ).ask()
    return model_name


def setup_hparams():
    """Configure number of epochs, learning rate, batch size, weight decay, etc."""
    
    hparams = {
        "epochs": 100,
        "learning_rate": 1e-3,
        "batch_size": 32,
        "weight_decay": 1e-2,
        "hidden_size": 128
    }

    while True:
        selected_hparam = questionary.select("Which hyperparameter do you wish to configure?",
                                             choices=[
                                                 f"Epochs ({hparams['epochs']})",
                                                 f"Learning rate ({hparams['learning_rate']})",
                                                 f"Batch size ({hparams['batch_size']})",
                                                 f"Weight decay ({hparams['weight_decay']})",
                                                 f"Hidden size ({hparams['hidden_size']})",
                                                 "Done"
                                                      ]
                                             ).ask()
    
        if selected_hparam == "Done":
            break

        bracket_start = selected_hparam.index("(") - 1
        selected_hparam = selected_hparam[:bracket_start].lower().replace(" ", "_")

        match selected_hparam:
            case "epochs":
                try:
                    epochs = int(input("Epochs: "))
                    hparams["epochs"] = epochs
                except:
                    raise ValueError("Number of epochs must be an integer")
            case "learning_rate":
                try:
                    learning_rate = float(input("Learning rate: "))
                    hparams["learning_rate"] = learning_rate
                except:
                    raise ValueError("Learning rate must be a floating point value")
            case "batch_size":
                try:
                    batch_size = int(input("Batch size: "))
                    hparams["batch_size"] = batch_size
                except:
                    raise ValueError("Batch size must be an integer")
            case "weight_decay":
                try:
                    weight_decay = float(input("Weight decay: "))
                    hparams["weight_decay"] = weight_decay
                except:
                    raise ValueError("Weight decay must be a floating point value")
            case "hidden_size":
                try:
                    hidden_size = int(input("Hidden layer size: "))
                    hparams["hidden_size"] = hidden_size
                except:
                    raise ValueError("Hidden size must be an integer")

    return hparams


if __name__ == '__main__':
    setup_hparams()
