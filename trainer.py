import sys
from datetime import datetime
import glob
import torch
import numpy as np
import itertools
import os
import time
from dataloader import get_dataloader
from comet_ml import Experiment


device = 'cuda' if torch.cuda.is_available() else 'cpu'

def extract_step_from_path(path):
    filename = os.path.split(path)[1]
    step = filename.split('_')[-1]
    step = step.split('.pt')[0]
    step = int(step)
    return step

def get_latest_checkpoint(template):
    glob_path = template.format(step='*')
    highest_step = -1

    for filepath in glob.glob(glob_path):
        step = extract_step_from_path(filepath)
        highest_step = max(highest_step, int(step))

    if highest_step == -1:
        return None
    path = template.format(step=highest_step)
    return path

class Trainer:
    def __init__(self,
        model=None,
        optimizer=None,
        scheduler=None,
        criterion=None,
        steps=1,
        accumulation_steps=1,
        dataloader_seed=None,
        dataset=None,
        batch_size=1,
        drop_last=True,
        augment_th=.5,
        normalize=True,
        augment=False,
        shuffle=True,
        roi_proposal_path=None,
        num_workers=0,
        prefetch_factor=None,
        checkpoint_template=None,
        experiment_tags=[],
    ):
        # constants
        self.dataset = dataset
        self.model = model.to(device)
        self.criterion = criterion.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.checkpoint_template = checkpoint_template
        self.accumulation_steps = accumulation_steps

        self.training_steps = steps
        self.total_samples = steps * batch_size

        # variables for checkpointing
        self.starting_step = None
        self.completed_steps = 0

        # variables for training
        self.current_step = -1
        self.should_step_optimizer = True
        self.should_step_scheduler = True
        self.should_zero_grad = True

        # dataloader
        self.shuffle = shuffle
        self.dataloader_seed = dataloader_seed
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.augment_th = augment_th
        self.normalize = normalize
        self.augment = augment
        self.shuffle = shuffle
        self.roi_proposal_path = roi_proposal_path
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor

        # track losses
        self.losses = []
        self.experiment = Experiment(
            project_name="faster-rcnn",
            workspace="ai6103",
            auto_param_logging=False,
            auto_metric_logging=False,
            auto_output_logging=False,
            auto_log_co2=False,
            log_code=False
        )
        for tag in experiment_tags:
            self.experiment.add_tag(tag)

    def train_one_step(self, step):
        batch = next(self.dataloader)
        if not (('x' in batch) and ('y' in batch)):
            raise Exception("batch must be a dict with 'x' and 'y' as keys")

        imgs, rois = batch['x']
        imgs = list(map(lambda x: x.to(device), imgs))
        if rois is None:
            pass
        else:
            rois = list(map(lambda x: x.to(device), rois))
        x = (imgs, rois)

        y = batch['y']
        y['class_ids'] = list(map(lambda x: x.to(device), y['class_ids']))
        y['bboxes'] = list(map(lambda x: x.to(device), y['bboxes']))
        batch_pred, y = self.model(x, y)

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

    def get_dataloader(self):
        number_of_batches = self.total_samples / self.batch_size
        if self.drop_last:
            number_of_batches = int(np.floor(number_of_batches))
        else:
            number_of_batches = int(np.ceil(number_of_batches))

        if self.training_steps > number_of_batches:
            raise ValueError("Insufficient number of batches for training")

        dataloader, dataloader_seed = get_dataloader(
            self.dataset,
            num_samples = self.total_samples,
            seed = self.dataloader_seed,
            skip = self.completed_steps,
            batch_size = self.batch_size,
            drop_last = self.drop_last,
            augment_th = self.augment_th,
            normalize = self.normalize,
            augment = self.augment,
            shuffle = self.shuffle,
            roi_proposal_path = self.roi_proposal_path,
            num_workers = self.num_workers,
            prefetch_factor = self.prefetch_factor,
        )
        return dataloader, number_of_batches, dataloader_seed

    def train(self):
        if self.completed_steps >= self.training_steps:
            raise Exception("Number of completed steps >= Number of training steps")

        '''
        check that sufficient number of batches can be drawn
        number of batches = number_of_samples / batch_size
        floor if drop last, ceil if not drop last
        number of steps should be <= number of batches
        '''
        self.starting_step = self.completed_steps
        dataloader, number_of_batches, seed = self.get_dataloader()
        self.dataloader = dataloader

        str_template = (
            '--- Training ---\n'
            'Number of steps: {num_steps}\n'
            'Start step: {start_step}\n'
            'Number of batches: {num_batch}\n'
            'Drop last: {drop_last}\n'
            '----------------'
        )
        str_formatted = str_template.format(
            num_steps=self.training_steps,
            drop_last=self.drop_last,
            num_batch=number_of_batches,
            start_step=self.completed_steps,
        )

        self.experiment.log_text(str_formatted)
        print(str_formatted)

        sys.stdout.flush()

        self.before_train()
        self.zero_grad(None)
        self.model.train()

        for step in range(self.completed_steps, self.training_steps):
            self.current_step = step
            self.before_step(step)

            loss = self.train_one_step(step)
            sys.stdout.flush()

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
        loss.backward()

        # Check for exploding gradients *before* zeroing them out
        for i, param in self.model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm()
                #print(f"Gradient norm for {i}: {grad_norm}")
                # Check for NaNs or Infs in gradients
                if torch.isnan(param.grad).any():
                    print(f"NaN detected in gradients of {i}: {grad_norm}")
                if torch.isinf(param.grad).any():
                    print(f"Inf detected in gradients of {i}: {grad_norm}")

        # Optional: Apply gradient clipping to prevent explosion
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)


    def before_train(self):
        pass

    def save_and_overwrite(self, save_step):
        # save checkpoint
        save_path = self.checkpoint_template.format(step=save_step)
        self.save_state(save_path)
        if not os.path.exists(save_path):
            msg = "Failed to save {}".format(save_path)
            #self.experiment.log_text(msg)
            print(msg)
            return None
        else:
            msg = "Saved {}".format(save_path)
            #self.experiment.log_text(msg)
            self.experiment.log_asset(save_path)
            print(msg)
        # remove other checkpoints once saved
        glob_path = self.checkpoint_template.format(step='*')
        for path in glob.glob(glob_path):
            step = extract_step_from_path(path)
            if step == save_step:
                continue
            os.remove(path)
        return save_path

    def after_train(self):
        path = self.save_and_overwrite(self.training_steps)

    def after_step(self, step, loss):
        if loss.isnan():
            path = self.checkpoint_template.format(step=step) + '.nan'
            self.save_state(path)
            msg = "step {} Loss is NaN. Check for issues? exiting".format(step)
            #self.experiment.log_text(msg)
            self.experiment.log_asset(path)
            print(msg)
            sys.exit(1)

        s = step+1
        now = datetime.now()
        now = now.ctime()
        msg = "{} - step {} | loss = {:.3f}".format(now, step, loss.item())
        print(msg)
        sys.stdout.flush()
        self.experiment.log_metric('train_loss', loss.item(), step=step)

        #if (s in [1, 1000, 2000]) or (s % 5000 == 0):
        if (s % 10000 == 0):
            path = self.save_and_overwrite(step)

    def load_latest_checkpoint(self):
        path = get_latest_checkpoint(self.checkpoint_template)
        if path is None:
            return
        self.load_state(path)

    def before_step(self, step):
        accumulate = (step + 1) % self.accumulation_steps == 0
        self.should_zero_grad = accumulate
        self.should_step_optimizer = accumulate
        self.should_step_scheduler = accumulate

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
