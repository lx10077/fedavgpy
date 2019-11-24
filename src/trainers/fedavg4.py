from src.trainers.base import BaseTrainer
from src.models.model import choose_model
from src.models.worker import LrdWorker
from src.optimizers.gd import GD
import numpy as np
import torch


criterion = torch.nn.CrossEntropyLoss()


class FedAvg4Trainer(BaseTrainer):
    """
    Scheme I and Scheme II, based on the flag of self.simple_average
    """
    def __init__(self, options, dataset):
        model = choose_model(options)
        self.move_model_to_gpu(model, options)

        self.optimizer = GD(model.parameters(), lr=options['lr'], weight_decay=options['wd'])
        self.num_epoch = options['num_epoch']
        worker = LrdWorker(model, self.optimizer, options)
        super(FedAvg4Trainer, self).__init__(options, dataset, worker=worker)
        self.prob = self.compute_prob()

    def train(self):
        print('>>> Select {} clients per round \n'.format(self.clients_per_round))

        # Fetch latest flat model parameter
        self.latest_model = self.worker.get_flat_model_params().detach()

        for round_i in range(self.num_round):

            # Test latest model on train data
            self.test_latest_model_on_traindata(round_i)
            self.test_latest_model_on_evaldata(round_i)

            # Choose K clients prop to data size
            if self.simple_average:
                selected_clients, repeated_times = self.select_clients_with_prob(seed=round_i)
            else:
                selected_clients = self.select_clients(seed=round_i)
                repeated_times = None

            # Solve minimization locally
            solns, stats = self.local_train(round_i, selected_clients)

            # Track communication cost
            self.metrics.extend_commu_stats(round_i, stats)

            # Update latest model
            self.latest_model = self.aggregate(solns, repeated_times=repeated_times)
            self.optimizer.inverse_prop_decay_learning_rate(round_i)

        # Test final model on train data
        self.test_latest_model_on_traindata(self.num_round)
        self.test_latest_model_on_evaldata(self.num_round)

        # Save tracked information
        self.metrics.write()

    def compute_prob(self):
        probs = []
        for c in self.clients:
            probs.append(len(c.train_data))
        return np.array(probs)/sum(probs)

    def select_clients_with_prob(self, seed=1):
        num_clients = min(self.clients_per_round, len(self.clients))
        np.random.seed(seed)
        index = np.random.choice(len(self.clients), num_clients, p=self.prob)
        index = sorted(index.tolist())

        select_clients = []
        select_index = []
        repeated_times = []
        for i in index:
            if i not in select_index:
                select_clients.append(self.clients[i])
                select_index.append(i)
                repeated_times.append(1)
            else:
                repeated_times[-1] += 1
        return select_clients, repeated_times

    def aggregate(self, solns, **kwargs):
        averaged_solution = torch.zeros_like(self.latest_model)
        # averaged_solution = np.zeros(self.latest_model.shape)
        if self.simple_average:
            repeated_times = kwargs['repeated_times']
            assert len(solns) == len(repeated_times)
            for i, (num_sample, local_solution) in enumerate(solns):
                averaged_solution += local_solution * repeated_times[i]
            averaged_solution /= self.clients_per_round
        else:
            for num_sample, local_solution in solns:
                averaged_solution += num_sample * local_solution
            averaged_solution /= self.all_train_data_num
            averaged_solution *= (100/self.clients_per_round)

        # averaged_solution = from_numpy(averaged_solution, self.gpu)
        return averaged_solution.detach()

