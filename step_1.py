from models import RPN
from config import cfg_1, cfg_3
from train import train_rpn_net, generate_rpn_proposals
from dataset import VOCDataset
import pickle
import torch

state_dict_path = './outputs/step_1_state.pkl'
proposal_path = './outputs/step_1_proposals.pkl'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
dataset = VOCDataset('trainval')
model = RPN(cfg_1, cfg_3)

print(model)

print('step_1_train')
model = train_rpn_net(model, dataset, device=device)
print('step_1_train complete')

with open(state_dict_path, 'wb') as fp:
	state_dict = model.state_dict()
	pickle.dump(state_dict, fp)
	print('saved', state_dict_path)

print('step_1_generate')
proposals = generate_rpn_proposals(model1, dataset, device=device, topN=2000, nms_th=.7)
print('step_1_generate complete')

with open(proposal_path, 'wb') as fp:
	pickle.dump(proposals, fp)
	print('saved', proposal_path)

print('complete')
