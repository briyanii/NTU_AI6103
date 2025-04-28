import matplotlib.pyplot as plt
import seaborn as sns
import torch
from scipy.interpolate import interp1d
import numpy as np

template_path = 'outputs/checkpoint_step{step}_80000.pt'
xs = np.array(list(range(80000))
ys = {}
for i in range(4):
    path = template_path.format(step=i+1)
    state = torch.load(path, map_location='cpu')
    losses = state['losses']
    losses = map(lambda l: l.item(), losses)
    ys[i+1] = torch.Tensor(list(losses)).numpy()

fig, axs = plt.subplots(2,2, figsize=(6,4))
axs = axs.T.ravel()

def moving_average(x, w):
    return np.convolve(x, np.ones(w), mode='valid')/w


for i in range(4):
    _ys = ys[i+1]
    ax = axs[i]
    interp_fn = interp1d(xs, _ys, kind='cubic')
    xnew = np.linspace(xs.min(), xs.max(), 100)
    ynew = interp_fn(xnew)

    window_size = 1000
    smoothed_loss = moving_average(_ys, window_size)
    smoothed_steps = xs[window_size-1:]
    ax.plot(xs, _ys, alpha=.2, c='blue', label='raw')
    ax.plot(smoothed_steps, smoothed_loss, c='blue', label='smooth')
    ax.set_title(f'Step{i+1}')
    ax.set_xticks([0, 40000, 80000])
    ax.set_xlim(0, 80000)
    if i in [0,2]:
        ax.set_ylim(0,6)
    else:
        ax.set_ylim(0,2)

fig.suptitle('training loss vs step')
fig.tight_layout()
fig.savefig('outputs/training_loss_curve.png')