import tqdm
import math
import matplotlib.pyplot as plt
from torchsummary import summary
from utils import *
from models import *

## Define network parameters
params = {}  # initialize parameters
params['filters'] = 8  # 'width' of the network - how much brain power it has per layer
params['layers'] = 2  # number of layers in the NN
params['dataset'] = 1 # dataset with random inputs and linear outputs
params['input length'] = 2  # dimensionality of the input
params['batch_size'] = 10  # number of training examples per batch
params['epochs'] = 10  # how many times to cycle through the training data (maximum)
params['average_over'] = 5 # how many epochs to average over for convergence testing
params['train_margin'] = 1e-4 # convergence flag
params['activation'] = 1  # type of activation function 1=ReLU, 2=Gaussian Kernel (experimental)
params['GPU'] = 1  # toggle for GPU evaluation
params['dataset seed'] = 1 # random seed for dataset generation

dataset_sizes = np.array((10000)) # range of dataset sizes to check
try:
    n_runs = len(dataset_sizes)
except:
    dataset_sizes = np.expand_dims(dataset_sizes,0) # this is necessary if we only are training at one dataset size
    n_runs = 1

tr_loss_record = np.zeros((n_runs,params['epochs']+1))
te_loss_record = np.zeros((n_runs,params['epochs']+1))

## run program
if __name__ == '__main__':

    for run in range(n_runs): # typically indexing over different dataset sizes
        params['dataset size'] = dataset_sizes[run]  # 604  # the maximum number of elements taken in the dataset
        dir_name = "dataset=%d_dataset_size=%d_filters=%d_layers=%d_activation=%d" %\
                   (params['dataset'],params['dataset size'], params['filters'], params['layers'], params['activation'])  # directory where logfiles will be saved

        if params['GPU'] == 1:
            backends.cudnn.benchmark = True  # auto-optimizes certain backend processes

        # initialize the model
        model = linear_net(params)
        # build dataset
        tr, te = get_dataloaders(params)
        # initialize the optimizer
        optimizer = optim.Adam(model.parameters(), amsgrad=True)
        #load checkpoint, if any exists
        model, optimizer, prev_epoch = load_checkpoint(model, optimizer, dir_name, params['GPU'], 0)

        if params['GPU'] == 1:
            model.to(torch.device("cuda:0"))
            print(summary(model, (1,params['input length']))) # only works on GPU for some reason

        # start training!
        err_tr_hist = [] # history of training losses
        err_te_hist = [] # history of test losses
        converged = 0 # convergence flag
        epoch = prev_epoch # if the model was reloaded, pick-up where it left off
        while (converged == 0) and (epoch <= (params['epochs']+prev_epoch)):#(epoch in tqdm.tqdm(range(prev_epoch, params['epochs']+prev_epoch))):
            err_tr = []
            model.train(True)
            for i, train_data in enumerate(tr): # get training loss batch-by-batch
                loss = get_loss(train_data, params, model)
                tr_loss_record[run,epoch-prev_epoch] = loss.cpu().detach().numpy() # multi-run loss record
                err_tr.append(loss.data) # record the loss

                optimizer.zero_grad() # run the optimizer
                loss.backward()
                optimizer.step()

            # also evaluate on the test dataset!
            err_te = []
            model.train(False)
            with torch.no_grad(): # we won't need gradients! no training just testing
                for i, test_data in enumerate(te): # get test loss batch-by-batch
                    loss = get_loss(test_data, params, model)
                    te_loss_record[run, epoch-prev_epoch] = loss.cpu().detach().numpy()
                    err_te.append(loss.data) # record the loss

            err_tr_hist.append(torch.mean(torch.stack(err_tr))) # record losses
            err_te_hist.append(torch.mean(torch.stack(err_te)))

            converged = auto_convergence(params, epoch - prev_epoch, torch.stack(err_tr_hist),torch.stack(err_te_hist)) # check convergence criteria
            epoch += 1
            # print outputs
            print('epoch={}; train loss={:.5f}; test loss={:.5f}'.format(epoch + prev_epoch, torch.mean(torch.stack(err_tr)), torch.mean(torch.stack(err_te)))) # print outputs



        # save checkpoint
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, 'ckpts/'+dir_name[:])

    output_dict = {} # save loss on all the runs
    output_dict['params']=params
    output_dict['train loss'] = tr_loss_record
    output_dict['test loss'] = te_loss_record

    np.save('outputs/'+dir_name[:],output_dict)

    # to load this dict x=np.load(dir_name + '_outputs.npy',allow_pickle=True)
    # then x=x.item()