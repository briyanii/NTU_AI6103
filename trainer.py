import torch
import numpy as np
import itertools
import os
import time

'''
Dataloader which draws randomly w/ replacement from a dataset
skip and seed allows for resuming
'''
def get_dataloader(dataset, num_samples, seed=None, collate_fn=None, skip=0, batch_size=1, drop_last=False):
    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)
    generator_seed = generator.initial_seed()

    sampler = torch.utils.data.RandomSampler(dataset, replacement=True, num_samples=num_samples, generator=generator)
    batch_sampler = torch.utils.data.BatchSampler(sampler, batch_size=batch_size, drop_last=drop_last)
    dataloader = torch.utils.data.DataLoader(dataset, batch_sampler=batch_sampler, collate_fn=collate_fn)
    sliced_dataloader = itertools.islice(dataloader, skip, None)

    return sliced_dataloader, generator_seed

class Trainer:
    def __init__(self, 
        model=None, optimizer=None, scheduler=None, criterion=None, steps=None,
        dataset=None, dataloader_seed=None, dataloader_batchsize=1, dataloader_droplast=True, dataloader_samples=1,
        collate_fn=None, accumulation_steps=1,
    ):
        # constants
        self.dataset = dataset
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion

        self.dataloader_seed = dataloader_seed
        self.batch_size = dataloader_batchsize
        self.drop_last = dataloader_droplast
        self.total_samples = dataloader_samples
        self.collate_fn = collate_fn

        self.accumulation_steps = accumulation_steps

        self.training_steps = steps

        # variables for checkpointing
        self.starting_step = None
        self.completed_steps = 0

        # variables for training
        self.current_step = -1
        self.should_step_optimizer = True
        self.should_step_scheduler = True
        self.should_zero_grad = True

        # track losses
        self.losses = []


    def train_one_step(self, step):
        batch = next(self.dataloader)
        x = batch['x']
        y = batch['y']
        if not (('x' in batch) and ('y' in batch)):
            raise Exception("batch must be a dict with 'x' and 'y' as keys")

        if x.isnan().any():
            print("NaN found in inputs")

        batch_pred = self.model(x)

        loss = self.criterion(batch_pred, y)
        loss = loss / self.accumulation_steps

        return loss

    def zero_grad(self, step):
        if self.should_zero_grad:
            self.optimizer.zero_grad()

    def optimizer_step(self, step):
        if self.should_step_optimizer:
            self.optimizer.step()

    def scheduler_step(self, step):
        if self.should_step_scheduler:
            self.scheduler.step()

    def train(self):
        if self.completed_steps >= self.training_steps:
            raise Exception("Number of completed steps >= Number of training steps")

        '''
        check that sufficient number of batches can be drawn
        number of batches = number_of_samples / batch_size
        floor if drop last, ceil if not drop last
        number of steps should be <= number of batches
        '''

        number_of_batches = self.total_samples / self.batch_size
        if self.drop_last:
            number_of_batches = int(np.floor(number_of_batches))
        else:
            number_of_batches = int(np.ceil(number_of_batches))

        if self.training_steps > number_of_batches:
            raise ValueError("Insufficient number of batches for training")

        self.dataloader, dataloader_seed = get_dataloader(
            self.dataset,
            self.total_samples,
            collate_fn=self.collate_fn,
            skip=self.completed_steps,
            seed=self.dataloader_seed,
            batch_size=self.batch_size,
            drop_last=self.drop_last,
        )
        self.dataloader_seed = dataloader_seed
        self.starting_step = self.completed_steps

        str_template = (
            '--- Training ---\n'
            'Number of steps: {num_steps}\n'
            'Start step: {start_step}\n'
            'Number of batches: {num_batch}\n'
            'Drop last: {drop_last}\n'
            '----------------'
        )
        print(str_template.format(
            num_steps=self.training_steps,
            drop_last=self.drop_last,
            num_batch=number_of_batches,
            start_step=self.completed_steps,
        ))

        self.before_train()
        self.zero_grad(None)
        for step in range(self.completed_steps, self.training_steps):
            self.current_step = step
            self.before_step(step)

            loss = self.train_one_step(step)
            self.losses.append(loss)
            self.backpropagate(loss)
            self.optimizer_step(step)
            self.scheduler_step(step)
            self.zero_grad(step)

            self.completed_steps = step + 1
            self.after_step(step, loss)
        self.after_train()

        del self.dataloader

    def save_state(self, path):
        state = {
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'model': self.model.state_dict(),
            'completed_steps': self.completed_steps,
            'starting_step': self.starting_step,
            'training_steps': self.training_steps,
            'dataloader_seed': self.dataloader_seed,
            'losses': self.losses,
        }
        torch.save(state, path)
        return state

    def load_state(self, path):
        if not os.path.exists(path):
            return None

        state = torch.load(path)

        self.optimizer.load_state_dict(state['optimizer'])
        self.model.load_state_dict(state['model'])
        self.scheduler.load_state_dict(state['scheduler'])
        self.completed_steps = state['completed_steps']
        self.dataloader_seed = state['dataloader_seed']
        self.losses = state['losses']
        self.starting_step = None

        return state

    def backpropagate(self, loss):
        # should be changed if multi loss, etc
        loss.backward()

        # Check for exploding gradients *before* zeroing them out
        for i, param in enumerate(self.model.parameters()):
            if param.grad is not None:
                #grad_norm = param.grad.norm()
                #print(f"Gradient norm for {i}: {grad_norm}")
                # Check for NaNs or Infs in gradients
                if torch.isnan(param.grad).any():
                    print(f"NaN detected in gradients of {i}")
                if torch.isinf(param.grad).any():
                    print(f"Inf detected in gradients of {i}")

        # Optional: Apply gradient clipping to prevent explosion
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)

    def before_train(self):
        pass

    def after_train(self):
        pass

    def after_step(self, step, loss):
        pass

    def before_step(self, step):
        accumulate = (step + 1) % self.accumulation_steps == 0
        self.should_zero_grad = accumulate
        self.should_step_optimizer = accumulate
        self.should_step_scheduler = True

'''
class ExampleTrainerImpl(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def before_step(self, step):
        self.should_step_optimizer = True
        self.should_step_scheduler = True
        self.should_zero_grad = True
        print('before step', step)

    def after_step(self, step, loss):
        print('after step', step, loss)

'''
