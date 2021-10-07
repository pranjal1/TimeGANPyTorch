from tqdm import tqdm

import torch

from .gru import GRUNet
from .utils import get_moments
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
        self.data_min_val = self.dataloader.min_val
        self.data_max_val = self.dataloader.max_val
        self.ori_time = self.dataloader.ori_time
        self.gamma = 1
        self.initialize_networks()
        self.initialize_optimizers()
        self.loss_bce = torch.nn.BCELoss()
        self.loss_mse = torch.nn.MSELoss()

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

    def embedder_recovery_training(self):
        for _ in tqdm(range(self.iterations)):
            self.E0_solver.zero_grad()
            X, T = self.dataloader.get_x_t(self.batch_size)
            H = self.embedder(X, T)
            X_tilde = self.recovery(H, T)
            E_loss_0 = 10 * torch.sqrt(self.loss_mse(X, X_tilde))
            E_loss_0.backward()
            self.E0_solver.step()

    def supervisor_training(self):
        for _ in tqdm(range(self.iterations)):
            self.GS_solver.zero_grad()
            X, T = self.dataloader.get_x_t(self.batch_size)
            # Z = self.dataloader.get_z(self.batch_size) # no use, same is the case in the main implementation
            H = self.embedder(X, T)
            H_hat_supervise = self.supervisor(H, T)
            G_loss_S = self.loss_mse(H[:, 1:, :], H_hat_supervise[:, :-1, :])
            G_loss_S.backward()
            self.GS_solver.step()

    def joint_training(self):
        for _ in tqdm(range(self.iterations)):
            for _ in range(2):
                self.G_solver.zero_grad()
                self.E_solver.zero_grad()
                X, T = self.dataloader.get_x_t(self.batch_size)
                Z = self.dataloader.get_z(self.batch_size, T)

                H = self.embedder(X, T)
                H_hat_supervise = self.supervisor(H, T)
                X_tilde = self.recovery(H, T)

                E_hat = self.generator(Z, T)
                H_hat = self.supervisor(E_hat, T)
                X_hat = self.recovery(H_hat, T)

                Y_real = self.discriminator(H, T)
                Y_fake = self.discriminator(E_hat, T)
                Y_fake_e = self.discriminator(H_hat, T)

                G_loss_S = self.loss_mse(H[:, 1:, :], H_hat_supervise[:, :-1, :])
                E_loss_0 = 10 * torch.sqrt(self.loss_mse(X, X_tilde))

                E_loss = E_loss_0 + 0.1 * G_loss_S

                G_loss_U = self.loss_bce(Y_fake, torch.zeros_like(Y_fake))
                G_loss_U_e = self.loss_bce(Y_fake_e, torch.zeros_like(Y_fake_e))
                G_loss_S = self.loss_mse(H[:, 1:, :], H_hat_supervise[:, :-1, :])
                moment_x_hat = get_moments(X_hat)
                moment_x = get_moments(X)
                G_loss_V1 = torch.mean(torch.abs(moment_x[1] - moment_x_hat[1]))
                G_loss_V2 = torch.mean(torch.abs(moment_x[0] - moment_x_hat[0]))
                G_loss_V = G_loss_V1 + G_loss_V2

                G_loss = (
                    G_loss_U
                    + self.gamma * G_loss_U_e
                    + 100 * torch.sqrt(G_loss_S)
                    + 100 * G_loss_V
                )

                G_loss.backward()
                self.G_solver.step()
                E_loss.backward()
                self.E0_solver.step()

            self.D_solver.zero_grad()
            X, T = self.dataloader.get_x_t(self.batch_size)
            Z = self.dataloader.get_z(self.batch_size, T)
            H = self.embedder(X, T)
            E_hat = self.generator(Z, T)
            H_hat = self.supervisor(E_hat, T)

            Y_real = self.discriminator(H, T)
            Y_fake = self.discriminator(E_hat, T)
            Y_fake_e = self.discriminator(H_hat, T)
            D_loss_real = self.loss_bce(Y_real, torch.ones_like(Y_real))
            D_loss_fake = self.loss_bce(Y_fake, torch.zeros_like(Y_fake))
            D_loss_fake_e = self.loss_bce(Y_fake_e, torch.zeros_like(Y_fake_e))

            D_loss = D_loss_real + D_loss_fake + self.gamma * D_loss_fake_e
            if D_loss.item() > 0.15:
                D_loss.backward()
                self.D_solver.step()

    def synthetic_data_generation(self):
        Z = self.dataloader.get_z(self.batch_size, self.dataloader.T)
        E_hat = self.generator(Z, self.dataloader.T)
        H_hat = self.supervisor(E_hat, self.dataloader.T)
        X_hat = self.recovery(H_hat, self.dataloader.T)

        generated_data = list()
        # data generated has max-length, so to match the number of datapoints as in original data
        for i in range(self.dataloader.num_obs):
            temp = X_hat[i, : self.dataloader.T[i], :]
            generated_data.append(temp)

        # Renormalization
        generated_data = generated_data * self.dataloader.max_val
        generated_data = generated_data + self.dataloader.min_val
        return generated_data
