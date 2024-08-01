import copy

import torch
from torchdyn.core import NeuralODE

# from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid, save_image
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import datasets, transforms
import numpy as np
import random

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

class torch_wrapper(torch.nn.Module):
    """Wraps model to torchdyn compatible format."""

    def __init__(self, model, classes):
        super().__init__()
        self.model = model
        self.classes = classes

    def forward(self, t, x, *args, **kwargs):
        #print((self.classes.to(x.device) * torch.ones_like(t).to(x.device)).int().shape, x.shape, t.shape)
        return self.model(t, x, (self.classes.to(x.device) * torch.ones_like(t).to(x.device)).int())

def generate_conditional_samples(model, parallel, savedir, step, net_="normal"):
    """Save 64 generated images (8 x 8) for sanity check along training.

    Parameters
    ----------
    model:
        represents the neural network that we want to generate samples from
    parallel: bool
        represents the parallel training flag. Torchdyn only runs on 1 GPU, we need to send the models from several GPUs to 1 GPU.
    savedir: str
        represents the path where we want to save the generated images
    step: int
        represents the current step of training
    """
    model.eval()

    classes = torch.randint(0,10,(64,))
    model_ = torch_wrapper(model, classes)
    if parallel:
        # Send the models from GPU to CPU for inference with NeuralODE from Torchdyn
        model_ = model_.module.to(device)

    node_ = NeuralODE(model_, solver="euler", sensitivity="adjoint")
    with torch.no_grad():
        traj = node_.trajectory(
            torch.randn(64, 3, 32, 32, device=device),
            t_span=torch.linspace(0, 1, 100, device=device),
        )
        traj = traj[-1, :].view([-1, 3, 32, 32]).clip(-1, 1)
        traj = traj / 2 + 0.5
    save_image(traj, savedir + f"{net_}_generated_FM_images_step_{step}.png", nrow=8)

    model.train()

def generate_samples(model, parallel, savedir, step, net_="normal"):
    """Save 64 generated images (8 x 8) for sanity check along training.

    Parameters
    ----------
    model:
        represents the neural network that we want to generate samples from
    parallel: bool
        represents the parallel training flag. Torchdyn only runs on 1 GPU, we need to send the models from several GPUs to 1 GPU.
    savedir: str
        represents the path where we want to save the generated images
    step: int
        represents the current step of training
    """
    model.eval()

    model_ = copy.deepcopy(model)
    if parallel:
        # Send the models from GPU to CPU for inference with NeuralODE from Torchdyn
        model_ = model_.module.to(device)

    node_ = NeuralODE(model_, solver="euler", sensitivity="adjoint")
    with torch.no_grad():
        traj = node_.trajectory(
            torch.randn(64, 3, 32, 32, device=device),
            t_span=torch.linspace(0, 1, 100, device=device),
        )
        traj = traj[-1, :].view([-1, 3, 32, 32]).clip(-1, 1)
        traj = traj / 2 + 0.5
    save_image(traj, savedir + f"{net_}_generated_FM_images_step_{step}.png", nrow=8)

    model.train()


def ema(source, target, decay):
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key].data.copy_(
            target_dict[key].data * decay + source_dict[key].data * (1 - decay)
        )


def infiniteloop(dataloader):
    while True:
        for x, y in iter(dataloader):
            yield x, y

class ClassSeparatedCIFAR10(Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None):
        self.dataset = datasets.CIFAR10(root, train=train, download=True, 
                                        transform=transform, target_transform=target_transform)
        self.data_by_class = self._separate_by_class()

    def _separate_by_class(self):
        data_by_class = {i: [] for i in range(10)}
        for idx, (_, target) in enumerate(self.dataset):
            data_by_class[target].append(idx)
        return data_by_class

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)

class NonOverlappingClassSampler(Sampler):
    def __init__(self, data_source, batch_size):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_classes = 10
        self.indices = list(range(len(data_source)))
        
    def __iter__(self):
        for i in range(len(self.data_source)):
            classes = np.random.permutation(self.num_classes)
            group1, group2 = classes[:self.num_classes//2], classes[self.num_classes//2:]

            indicies1 = sum([self.data_source.data_by_class[c] for c in group1], [])
            indicies2 = sum([self.data_source.data_by_class[c] for c in group2], [])

            yield random.sample(indicies1, self.batch_size) + random.sample(indicies2, self.batch_size)


    def __len__(self):
        return len(self.data_source) // (self.batch_size * 2)

element_max = np.vectorize(max)

def calculate_P_vectorized(costs, u1, u2, u3):

    P = np.zeros_like(costs)
    # Calculate outer product for 3 vectors (every element times every element)
    outer_product = np.einsum('i,j,k->ijk', u1, u2, u3)
    # Update transport plan P
    P = costs * outer_product

    return P

def update_u_vectorized(costs, s, a, u1, u2):
    u = np.zeros(costs.shape[s])
    all_s = [0,1,2]
    all_s.remove(s)
    for i_s in range(costs.shape[s]):
        # Calculate denominator
        outer_product = np.outer(u1,u2)
        slc = [slice(None)] * 3
        slc[s] = i_s
        cost_slice = costs[tuple(slc)]
        denom = np.sum(outer_product * cost_slice)
        u[i_s] = a[i_s] / element_max(denom, 1e-300) # avoid divided by zero

    return u

def sinkhorn3(costs, a, b, c, epsilon, precision):
    a = a.reshape((costs.shape[0], 1))
    b = b.reshape((costs.shape[1], 1))
    c = c.reshape((costs.shape[2], 1))
    K = np.exp(-costs/epsilon)

    # initialization
    u1 = np.ones((K.shape[0],))
    u2 = np.ones((K.shape[1],))
    u3 = np.ones((K.shape[2],))
    P = calculate_P_vectorized(K, u1, u2, u3)

    #p_norm = np.trace(P.T @ P)

    #while True:
    for _ in range(100):
        u1 = update_u_vectorized(K, 0, a, u2, u3)
        u2 = update_u_vectorized(K, 1, b, u1, u3)
        u3 = update_u_vectorized(K, 2, c, u1, u2)
        P = calculate_P_vectorized(K, u1, u2, u3)

        #plt.imshow(P * 200)
        #plt.show()

        #if abs((np.trace(P.T @ P) - p_norm)/p_norm) < precision:
        #    break
        #p_norm = np.trace(P.T @ P)
    return P

def vectorized_heron_formula(points1, points2, points3):
    # Calculate the differences between points
    diff1 = points2 - points1
    diff2 = points3 - points2
    diff3 = points1 - points3

    # Calculate the squared lengths of the sides
    a_squared = np.sum(diff1**2, axis=-1)
    b_squared = np.sum(diff2**2, axis=-1)
    c_squared = np.sum(diff3**2, axis=-1)

    # Calculate the lengths of the sides
    a = np.sqrt(a_squared)
    b = np.sqrt(b_squared)
    c = np.sqrt(c_squared)

    # Calculate the semi-perimeter
    s = (a + b + c) / 2

    # Calculate the area using Heron's formula
    area = np.sqrt(s * (s - a) * (s - b) * (s - c))

    return area

def calculate_perimeter(points1, points2, points3):
    # Calculate the differences between points
    diff1 = points2 - points1
    diff2 = points3 - points2
    diff3 = points1 - points3

    # Calculate the squared lengths of the sides
    a_squared = np.sum(diff1**2, axis=-1)
    b_squared = np.sum(diff2**2, axis=-1)
    c_squared = np.sum(diff3**2, axis=-1)

    # Calculate the lengths of the sides
    a = np.sqrt(a_squared)
    b = np.sqrt(b_squared)
    c = np.sqrt(c_squared)

    # Calculate the perimeter
    p = (a + b + c) 

    # Calculate the area using Heron's formula    
    return p

def calculate_house_cost(points1, points2, points3):
    # Calculate the differences between points
    diff1 = points2 - points1
    diff2 = points3 - points2
    diff3 = points1 - points3

    # Calculate the squared lengths of the sides
    a_squared = np.sum(diff1**2, axis=-1)
    b_squared = np.sum(diff2**2, axis=-1)
    c_squared = np.sum(diff3**2, axis=-1)

    # Calculate the lengths of the sides
    a = np.sqrt(a_squared)
    b = np.sqrt(b_squared)
    c = np.sqrt(c_squared)

    # Calculate the perimeter
    p = (a + (b ** 2)  + c) 

    # Calculate the area using Heron's formula    
    return p

def get_cost_matrix(points1, points2, points3, cost_fn):
    # Reshape points to enable 
    points1 = points1.reshape(points1.shape[0], -1)
    points2 = points2.reshape(points2.shape[0], -1)
    points3 = points3.reshape(points3.shape[0], -1)
    p1 = points1[:, np.newaxis, np.newaxis, :]
    p2 = points2[np.newaxis, :, np.newaxis, :]
    p3 = points3[np.newaxis, np.newaxis, :, :]

    # Calculate the cost matrix using vectorized operations
    if cost_fn == 'area':
        cost = vectorized_heron_formula(p1.astype(np.float64), p2.astype(np.float64), p3.astype(np.float64))
    elif cost_fn == 'perimeter':
        cost = calculate_perimeter(p1.astype(np.float64), p2.astype(np.float64), p3.astype(np.float64))
    elif cost_fn == 'house':
        cost = calculate_house_cost(p1.astype(np.float64), p2.astype(np.float64), p3.astype(np.float64))
    
    return cost * (200 / np.mean(cost))

def sample_plan(x0, x1, x2, y0, y1, y2, cost_fn='perimeter'):
    cost_matrix = get_cost_matrix(x0, x1, x2, cost_fn)
    a = np.ones(x0.shape[0]) / x0.shape[0]
    b = np.ones(x1.shape[0]) / x1.shape[0]
    c = np.ones(x2.shape[0]) / x2.shape[0]
    P1 = sinkhorn3(cost_matrix, a, b, c, epsilon = 1, precision = 1e-3)
    p = P1.flatten()
    p = p / p.sum()
    if np.isnan(p).any():
        print('WARNING : NaN detected in P')
        print('P:', p)
        print('cost:', cost_matrix)
        p = np.nan_to_num(p)

    #p = p ** 5
    #p = p / np.sum(p)
    choices = np.random.choice(
        P1.shape[0] * P1.shape[1] * P1.shape[2], p=p, size=P1.shape[0], replace=True
    )
    i, j, k = choices // (P1.shape[0] * P1.shape[1]), (choices // P1.shape[0]) % P1.shape[1], choices % P1.shape[1]
    return (
            x0[i], 
            x1[j], 
            x2[k],
            y0[i] if y0 is not None else None,
            y1[j] if y1 is not None else None,
            y2[k] if y2 is not None else None,
        )

def sample_conditional_pt(x0, x1, t, sigma):
    t = t.reshape(-1, *([1] * (x0.dim() - 1)))
    mu_t = t * x1 + (1 - t) * x0
    epsilon = torch.randn_like(x0)
    return mu_t + sigma * epsilon