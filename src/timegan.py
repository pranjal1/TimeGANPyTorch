from tqdm import tqdm

import torch

from .gru import GRUNet
from .dataloader import TimeSeriesDataLoader


class RecurrentNetwork:
    def __init__(
        self,
        module_name,
        num_layers,
        input_dimension,
        hidden_dimension,
        output_dimension,
        max_seq_length,
    ) -> None:
        self.module_name = module_name
        assert self.module_name in ["gru", "lstm"]  # lstmLN is missing
        self.num_layers = num_layers
        self.input_dimension = input_dimension
        self.hidden_dimension = hidden_dimension
        self.output_dimension = output_dimension
        self.max_seq_length = max_seq_length
        self.network = GRUNet(
            self.input_dimension,
            self.hidden_dimension,
            self.output_dimension,
            self.num_layers,
            self.max_seq_length,
        )

    def __call__(self, X, T):
        out = self.network.forward(X, T)
        return out


class TimeGAN:
    def __init__(self, parameters):
        self.parameters = parameters

        # Network Parameters
        self.hidden_dim = self.parameters["hidden_dim"]
        self.num_layers = self.parameters["num_layer"]
        self.iterations = self.parameters["iterations"]
        self.batch_size = self.parameters["batch_size"]
        self.module_name = self.parameters["module"]
        self.sequence_length = self.parameters["sequence_length"]
        self.dataloader = TimeSeriesDataLoader("energy", self.sequence_length)
        self.max_seq_length = self.dataloader.max_seq_len
        self.ip_dimension = self.dataloader.dim
        self.gamma = 1
        self.initialize_networks()
        self.initialize_optimizers()

    def initialize_networks(self):
        self.embedder = RecurrentNetwork(
            "gru",
            self.num_layers,
            self.ip_dimension,
            self.hidden_dim,
            self.hidden_dim,
            self.max_seq_length,
        )
        self.recovery = RecurrentNetwork(
            "gru",
            self.num_layers,
            self.hidden_dim,
            self.hidden_dim,
            self.ip_dimension,
            self.max_seq_length,
        )

        self.discriminator = RecurrentNetwork(
            "gru",
            self.num_layers,
            self.hidden_dim,
            self.hidden_dim,
            1,
            self.max_seq_length,
        )
        self.generator = RecurrentNetwork(
            "gru",
            self.num_layers,
            self.ip_dimension,
            self.hidden_dim,
            self.hidden_dim,
            self.max_seq_length,
        )

        self.supervisor = RecurrentNetwork(
            "gru",
            self.num_layers - 1,
            self.hidden_dim,
            self.hidden_dim,
            self.hidden_dim,
            self.max_seq_length,
        )

    def initialize_optimizers(self):
        self.E0_solver_params = [
            *self.embedder.network.parameters(),
            *self.recovery.network.parameters(),
        ]
        self.E0_solver = torch.optim.Adam(
            self.E0_solver_params, lr=self.parameters["learning_rate"]
        )

        self.E_solver_params = [
            *self.embedder.network.parameters(),
            *self.recovery.network.parameters(),
        ]
        self.E_solver = torch.optim.Adam(
            self.E_solver_params, lr=self.parameters["learning_rate"]
        )

        self.D_solver_params = [
            *self.discriminator.network.parameters(),
        ]
        self.D_solver = torch.optim.Adam(
            self.D_solver_params, lr=self.parameters["learning_rate"]
        )

        self.G_solver_params = [
            *self.generator.network.parameters(),
            *self.supervisor.network.parameters(),
        ]
        self.G_solver = torch.optim.Adam(
            self.G_solver_params, lr=self.parameters["learning_rate"]
        )

        self.GS_solver_params = [
            *self.generator.network.parameters(),
            *self.supervisor.network.parameters(),
        ]
        self.GS_solver = torch.optim.Adam(
            self.GS_solver_params, lr=self.parameters["learning_rate"]
        )

    def discriminator_training(self):
        for _ in tqdm(self.iterations):
            self.D_solver.zero_grad()
            X, T = self.dataloader.get_x_t(self.batch_size)
            Z = self.dataloader.get_z(self.batch_size)
            H = self.embedder(X, T)
            E_hat = self.generator(Z, T)
            H_hat = self.supervisor(E_hat, T)

            Y_real = self.discriminator(H, T)
            Y_fake = self.discriminator(E_hat, T)
            Y_fake_e = self.discriminator(H_hat, T)
            D_loss_real = torch.nn.BCELoss(Y_real, torch.ones_like(Y_real))
            D_loss_fake = torch.nn.BCELoss(Y_fake, torch.ones_like(Y_fake))
            D_loss_fake_e = torch.nn.BCELoss(Y_fake_e, torch.ones_like(Y_fake_e))

            D_loss = D_loss_real + D_loss_fake + self.gamma * D_loss_fake_e
            D_loss.backward()
            self.D_solver.step()
