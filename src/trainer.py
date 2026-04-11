import mlx.nn as nn
import mlx.core as mx
import mlx.optimizers as optim
import mlx.data as dx 

import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

from .base_encoder_nn import unflatten
from .config import OptimizerConfig



class Trainer:

    def __init__(self, model: nn.Module, optim_config: OptimizerConfig, buffer: dx.Buffer, model_name: str, train_size: float = 0.7, batch_size: int = 256):
        self.model = model
        self.optimizer = optim_config.optimizer
        self.model_name = model_name

        self.train_lenght = int(len(buffer )* train_size)
        strain = time.time()
        self.train_buffer = dx.buffer_from_vector([buffer[i] for i in range(self.train_lenght)]) # @flag to optimize
        print("MLX train buffer init in ", time.time() - strain)
        stest = time.time()
        self.test_buffer = dx.buffer_from_vector([buffer[i] for i in range(self.train_lenght, len(buffer))])
        print("MLX test buffer init in ", time.time() - stest)
        self.batch_size = batch_size
        

    def _train(self, stream) -> float:
        return 0.
    
    def _test(self, stream) -> float:
        return 0.
    
    def epoch(self, n_epoch: int, save_dir: str, model_name: str, plot_chart: bool):
        os.makedirs(save_dir, exist_ok=True)
        loss_train_history, loss_test_history = [], []
        best_loss = mx.inf

        for e in range(1, n_epoch + 1):
            train_stream = (self.train_buffer.shuffle().to_stream().batch(self.batch_size))
            test_stream = (self.test_buffer.shuffle().to_stream().batch(self.batch_size))
            print(f"EPOCH {e} --- ")
            train_loss = self._train(train_stream)
            test_loss = self._test(test_stream)
            print(f"TRAIN LOSS {train_loss} ||| TEST LOSS {test_loss}")
            loss_train_history.append(train_loss)
            loss_test_history.append(test_loss)

            if best_loss > test_loss:
                self.model.save_weights(os.path.join(save_dir, model_name + ".safetensors"))
                best_loss = test_loss

        if plot_chart:
            fig = plt.figure(figsize=(9, 6))
            ax1 = fig.add_subplot(121)
            ax1.set_title(f"-- {model_name} -- Train Loss")
            ax1.plot(range(len(loss_train_history)), loss_train_history)
            ax1.grid()

            ax2 = fig.add_subplot(122)
            ax2.set_title(f"-- {model_name} -- Test Loss")
            ax2.plot(range(len(loss_test_history)), loss_test_history)
            ax2.grid()
            
            fig.savefig(os.path.join(save_dir, f"{self.model_name}-loss history"))
            plt.show()


class VolatilityTrainer(Trainer):

    def __init__(self, model, optim_config, buffer, categorical_columns, train_size = 0.7, batch_size = 256, scale = 10):
        self.scale = scale
        self.categorical_columns = categorical_columns
        super().__init__(model=model, optim_config=optim_config, buffer=buffer, train_size=train_size, batch_size=batch_size, model_name="Volatility_Estimator")

    @staticmethod
    def _MSE(model, target, data, categorical):
        pred = model(features=data, categorical_features=categorical)
        return ((target - pred)**2).mean().sqrt()

    def _train(self, stream):
        loss = nn.value_and_grad(self.model, self._MSE)
        batch_count = 0
        total_loss = 0.0
        self.model.train()

        for batch in tqdm(stream):
            target = mx.array(batch["target"]) * self.scale        
            categorical   = unflatten(self.categorical_columns, batch["categorical"])
            data   = mx.array(batch["data"])
            
            loss_value, grad = loss(model=self.model, target=target, data=data, categorical=categorical)
            
            grad, _ = optim.clip_grad_norm(grad, max_norm=1.0)
            self.optimizer.update(self.model, grad)
            mx.eval(self.model.parameters(), self.optimizer.state, loss_value)
            total_loss+= loss_value.item() / (self.scale**2)
                
            batch_count+=1


        return total_loss / batch_count

    def _test(self, stream):
        self.model.train(False)
        total_loss = 0.0
        batch_count = 0
        
        for batch in tqdm(stream):
            target = mx.array(batch["target"]) * self.scale       
            categorical   = unflatten(self.categorical_columns, batch["categorical"])
            data   = mx.array(batch["data"])
            pred = self.model(features=data, categorical_features=categorical)
            loss_value = ((target - pred)**2).mean().sqrt()
            total_loss+= loss_value.item() / (self.scale ** 2)# call eval
            batch_count+=1

        return total_loss / batch_count

class VAETrainer(Trainer):

    def __init__(self, model, optim_config, buffer, categorical_columns, train_size = 0.7, batch_size = 256):
        super().__init__(model=model, optim_config=optim_config, buffer=buffer, train_size=train_size, batch_size=batch_size, model_name="VAE")
        self.categorical_columns = categorical_columns

    @staticmethod
    def _VAE_LOSS(model, c_input, categorical_input):
        LAMBDA_CAT = 0.01
        BETA = 0.1
        (c_pred, categorical_pred), params = model(features=c_input, categorical_features=categorical_input)
        c_mse = nn.losses.mse_loss(c_pred, c_input, reduction="mean")
        categorical_loss = mx.array(0.0, dtype=mx.float32)
        for name in list(categorical_input.keys()):
            categorical_loss += nn.losses.cross_entropy(categorical_input[name], categorical_pred[name], reduction="mean")
        
        mu, logvar = params
        kl_loss = mx.mean(-0.5 * mx.sum(1. + logvar - mu**2 - mx.exp(logvar)))

        return c_mse + (categorical_loss*LAMBDA_CAT) + kl_loss * BETA
    
    def _train(self, stream):
        loss = nn.value_and_grad(self.model, self._VAE_LOSS)
        total_loss = 0.0
        batch_count = 0
        self.model.train()

        for batch in tqdm(stream):
            categorical   = unflatten(self.categorical_columns, batch["categorical"])
            data   = mx.array(batch["data"])        
            loss_value, grad = loss(model=self.model, c_input=data, categorical_input=categorical)        
            grad, _ = optim.clip_grad_norm(grad, max_norm=1.0)
            self.optimizer.update(self.model, grad)
            mx.eval(self.model.parameters(), self.optimizer.state, loss_value)
            total_loss+= loss_value.item() 
            batch_count+=1

        return total_loss / batch_count

    def _test(self, stream):
        self.model.train(False)
        total_loss = 0.0
        batch_count = 0
        
        for batch in tqdm(stream):
            data = mx.array(batch["data"]) 
            categorical   = unflatten(self.categorical_columns, batch["categorical"])
            loss = self._VAE_LOSS(model=self.model, c_input=data, categorical_input=categorical)
            total_loss+= loss.item()
            batch_count+=1

        return total_loss / batch_count

class DiscretizedVolTrainer(Trainer):

    def __init__(self, model, optim_config, buffer, categorical_columns, n_states = 3, train_size = 0.7, batch_size = 256):
        super().__init__(model=model, optim_config=optim_config, buffer=buffer, train_size=train_size, batch_size=batch_size, model_name='Volatility_Predictor')
        self.n_states = n_states
        self.categorical_columns = categorical_columns

    @staticmethod
    def _CEL(model, inputs, target):
        pred = model(**inputs)
        return nn.losses.cross_entropy(pred, target, reduction="mean")

    def _train(self, stream):
        self.model.train()
        loss = nn.value_and_grad(self.model, self._CEL)
        total_loss = 0.0
        batch_count = 0

        for batch in tqdm(stream):
            target = mx.array(batch["target"])       
            categorical   = unflatten(self.categorical_columns, batch["categorical"])
            data   = mx.array(batch["data"])
            loss_value, grad = loss(model=self.model, inputs={"features": data, "categorical_features": categorical}, target=target)

            grad, _ = optim.clip_grad_norm(grad, max_norm=1.0)
            self.optimizer.update(self.model, grad)
            mx.eval(self.model.parameters(), self.optimizer.state, loss_value)
            total_loss+= loss_value.item()

            batch_count+=1
        
        return total_loss / batch_count
    
    def _test(self, stream):
        self.model.train(False)
        total_loss, acc = 0.0, 0.0
        batch_count = 0
        
        for batch in stream:
            target = mx.array(batch["target"])   
            categorical   = unflatten(self.categorical_columns, batch["categorical"])
            data   = mx.array(batch["data"])
            pred = self.model(features=data, categorical_features=categorical)
            loss_value = nn.losses.cross_entropy(pred, target, reduction="mean")
            total_loss += loss_value.item()
            acc += ((pred.argmax(axis=-1) == target).sum()) / len(target)
            batch_count+=1
        print("TEST ACCURACY - ", (acc / batch_count)*100, "%")
        return loss_value / batch_count
