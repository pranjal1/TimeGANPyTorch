import os
import datetime
from tqdm import tqdm

import torch
from loguru import logger

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
        self.dataset = self.parameters["dataset"]
        self.dataloader = TimeSeriesDataLoader(self.dataset, self.sequence_length)
        self.max_seq_length = self.dataloader.max_seq_len
        self.ip_dimension = self.dataloader.dim
        self.data_min_val = self.dataloader.min_val
        self.data_max_val = self.dataloader.max_val
        self.ori_time = self.dataloader.T
        self.gamma = 1
        self.initialize_networks()
        self.initialize_optimizers()
        self.loss_bce = torch.nn.functional.binary_cross_entropy_with_logits()
        self.loss_mse = torch.nn.MSELoss()
        self.file_storage()

    def file_storage(self):
        ts = datetime.datetime.today().strftime("%Y-%m-%d-%H-%M-%S")
        log_dir = os.path.join(os.path.dirname(__file__), ("../logs_" + ts))
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)
        self.embedder_recovery_error_log = os.path.join(
            log_dir, "embedder_recovery_error_log.log"
        )
        self.supervisor_error_log = os.path.join(log_dir, "supervisor_error_log.log")
        self.joint_generator_error_log = os.path.join(
            log_dir, "joint_generator_error_log.log"
        )
        self.joint_embedder_recovery_error_log = os.path.join(
            log_dir, "joint_embedder_recovery_error_log.log"
        )
        self.joint_discriminator_error_log = os.path.join(
            log_dir, "joint_discriminator_error_log.log"
        )
        log_files = [
            self.embedder_recovery_error_log,
            self.supervisor_error_log,
            self.joint_generator_error_log,
            self.joint_embedder_recovery_error_log,
            self.joint_discriminator_error_log,
        ]
        for f in log_files:
            with open(f, "w") as f:
                f.write("iteration,loss\n")
        self.model_save_path = os.path.join(log_dir, "model.pth")

    def save_model(self):
        logger.info("Saving model...")
        torch.save(
            {
                "embedder_state_dict": self.embedder.network.state_dict(),
                "recovery_state_dict": self.recovery.network.state_dict(),
                "supervisor_state_dict": self.supervisor.network.state_dict(),
                "generator_state_dict": self.generator.network.state_dict(),
                "discriminator_state_dict": self.discriminator.network.state_dict(),
            },
            self.model_save_path,
        )
        logger.info("Saving model complete!!!")

    def load_model(self, path):
        check_point = torch.load(path)
        self.embedder.network.load_state_dict(check_point["embedder_state_dict"])
        self.recovery.network.load_state_dict(check_point["recovery_state_dict"])
        self.supervisor.network.load_state_dict(check_point["supervisor_state_dict"])
        self.generator.network.load_state_dict(check_point["generator_state_dict"])
        self.discriminator.network.load_state_dict(
            check_point["discriminator_state_dict"]
        )

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
        logger.info("Traning Embedder and Recovery Networks...")
        for i in tqdm(range(self.iterations)):
            self.E0_solver.zero_grad()
            X, T = self.dataloader.get_x_t(self.batch_size)
            H = self.embedder(X, T)
            X_tilde = self.recovery(H, T)
            E_loss_0 = 10 * torch.sqrt(self.loss_mse(X, X_tilde))
            E_loss_0.backward()
            self.E0_solver.step()
            with open(self.embedder_recovery_error_log, "a") as f:
                f.write("{},{}".format(i, str(E_loss_0.item())))
                f.write("\n")
        logger.info("Traning Embedder and Recovery Networks complete")

    def supervisor_training(self):
        logger.info("Traning Supervisor Network...")
        for i in tqdm(range(self.iterations)):
            self.GS_solver.zero_grad()
            X, T = self.dataloader.get_x_t(self.batch_size)
            # Z = self.dataloader.get_z(self.batch_size) # no use, same is the case in the main implementation
            H = self.embedder(X, T)
            H_hat_supervise = self.supervisor(H, T)
            G_loss_S = self.loss_mse(H[:, 1:, :], H_hat_supervise[:, :-1, :])
            G_loss_S.backward()
            self.GS_solver.step()
            with open(self.supervisor_error_log, "a") as f:
                f.write("{},{}".format(i, str(G_loss_S.item())))
                f.write("\n")
        logger.info("Traning Supervisor Network complete")

    def joint_training(self):
        logger.info("Performing Joint Network training...")
        for i in tqdm(range(self.iterations)):
            for kk in range(2):
                X, T = self.dataloader.get_x_t(self.batch_size)
                Z = self.dataloader.get_z(self.batch_size, T)

                self.E_solver.zero_grad()
                H = self.embedder(X, T)
                H_hat_supervise = self.supervisor(H, T)
                X_tilde = self.recovery(H, T)

                G_loss_S = self.loss_mse(H[:, 1:, :], H_hat_supervise[:, :-1, :])
                E_loss_0 = 10 * torch.sqrt(self.loss_mse(X, X_tilde))

                E_loss = E_loss_0 + 0.1 * G_loss_S
                E_loss.backward()
                self.E0_solver.step()

                self.G_solver.zero_grad()
                H = self.embedder(X, T)
                H_hat_supervise = self.supervisor(H, T)

                E_hat = self.generator(Z, T)
                H_hat = self.supervisor(E_hat, T)
                X_hat = self.recovery(H_hat, T)

                Y_real = self.discriminator(H, T)
                Y_fake = self.discriminator(E_hat, T)
                Y_fake_e = self.discriminator(H_hat, T)

                G_loss_U = self.loss_bce(Y_fake, torch.zeros_like(Y_fake))
                G_loss_U_e = self.loss_bce(Y_fake_e, torch.zeros_like(Y_fake_e))
                G_loss_S = self.loss_mse(H[:, 1:, :], H_hat_supervise[:, :-1, :])

                # Two Momments
                G_loss_V1 = torch.mean(
                    torch.abs(
                        torch.sqrt(X_hat.var(dim=0, unbiased=False) + 1e-6)
                        - torch.sqrt(X.var(dim=0, unbiased=False) + 1e-6)
                    )
                )
                G_loss_V2 = torch.mean(torch.abs((X_hat.mean(dim=0)) - (X.mean(dim=0))))
                G_loss_V = G_loss_V1 + G_loss_V2

                G_loss = (
                    G_loss_U
                    + self.gamma * G_loss_U_e
                    + 100 * torch.sqrt(G_loss_S)
                    + 100 * G_loss_V
                )
                G_loss.backward()
                self.G_solver.step()

                with open(self.joint_generator_error_log, "a") as f:
                    f.write("{},{}".format(i * 2 + kk, str(G_loss.item())))
                    f.write("\n")
                with open(self.joint_embedder_recovery_error_log, "a") as f:
                    f.write("{},{}".format(i * 2 + kk, str(E_loss.item())))
                    f.write("\n")

            X, T = self.dataloader.get_x_t(self.batch_size)
            Z = self.dataloader.get_z(self.batch_size, T)
            self.D_solver.zero_grad()
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

            with open(self.joint_discriminator_error_log, "a") as f:
                f.write("{},{}".format(i, str(D_loss.item())))
                f.write("\n")
        logger.info("Joint Network training complete")

    def synthetic_data_generation(self):
        Z = self.dataloader.get_z(self.dataloader.num_obs, self.dataloader.T)
        E_hat = self.generator(Z, self.dataloader.T)
        H_hat = self.supervisor(E_hat, self.dataloader.T)
        X_hat = self.recovery(H_hat, self.dataloader.T)

        generated_data = list()
        # data generated has max-length, so to match the number of datapoints as in original data
        for i in range(self.dataloader.num_obs):
            temp = X_hat[i, : self.dataloader.T[i], :]
            generated_data.append(temp)

        # Renormalization
        generated_data_scaled = []
        for x in generated_data:
            x = x.cpu().detach().numpy()
            x = (x * self.dataloader.max_val) + self.dataloader.min_val
            generated_data_scaled.append(x)
        return generated_data_scaled

    def train(self):
        try:
            self.embedder_recovery_training()
            self.save_model()
            self.supervisor_training()
            self.save_model()
            self.joint_training()
            self.save_model()
        except KeyboardInterrupt:
            logger.error("KeyBoard Interrupt!")
            self.save_model()
