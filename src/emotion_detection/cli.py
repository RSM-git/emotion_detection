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
    answers = questionary.form(
            num_epochs = questionary.select("Number of epochs", choices=["50", "100", "150"]),
            learning_rate = questionary.select("Learning rate", choices=["1e-2", "1e-3", "1e-4"]),
            batch_size = questionary.select("Batch size", choices=["32", "64", "128"]),
            weight_decay = questionary.select("Weight decay", choices=["1e-1", "1e-2", "1e-3"])
    ).ask()

    print(answers)
    return answers


if __name__ == '__main__':
    setup_hparams()
