import torch
from data_preprocess import DataPreprocessor
from model import CustomCNN
from trainer import SentimentTrainer
from visualiser import Visualizer


class SAKAN:
    def train(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # Data preprocessing
        preprocessor = DataPreprocessor()
        train, test = preprocessor.load_data("../data/imdb_train_original.csv", "../data/imdb_test.csv")
        train_processed = preprocessor.process_data(train)
        test_processed = preprocessor.process_data(test)

        # Model init
        vocab_size = len(preprocessor.tokenizer)
        embed_dim = 32
        model = CustomCNN(vocab_size, embed_dim).to(device)

        # Load into tensors
        trainer = SentimentTrainer(model, device)
        trainer.prepare_data(train_processed.tokenized, train_processed.label,
                             test_processed.tokenized, test_processed.label)

        # Export model visual
        Visualizer.visualize_model(model, (trainer.batch_size, 256), vocab_size, device)

        # Train
        model_history = trainer.train(num_epochs=20)

        # Visualisation
        Visualizer.plot_history(model_history)

        # Save
        torch.save(model, './model_saves/model_complete.pth')
