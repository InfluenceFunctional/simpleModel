import matplotlib.pyplot as plt
from utils import *
from model import *
import glob

params = {}  # initialize parameters
params['run num'] = 0

# Simulation parameters
params['workdir'] = 'C:/Users\mikem\Desktop/simpleNetRuns' # TODO make this a directory on your local machine!
params['explicit run enumeration'] = False # if this is True, the next run be fresh, in directory 'run%d'%run_num, if false, regular behaviour. Note: only use this on fresh runs
params['GPU'] = 0  # toggle for GPU evaluation

# Define network parameters
params['model filters'] = 128  # AKA number of nodes, the 'width' of the network
params['model layers'] = 20 # number of layers in the NN, the 'depth' of the network
params['ensemble size'] = 1 # number of models in the ensemble (for >1, we can compute the variance in predictions from different models)
params['activation'] = 1  # type of activation function 1=ReLU, 2=Gaussian Kernel (experimental, expensive)

# dataset parameters
params['dataset'] = 1 # 1=dataset with random inputs and linear outputs, 2= not implemented - could load a dataset
params['input length'] = 1  # dimensionality of the input - must agree with what is loaded or created in toyFunction. Figures are set up for 1D
params['dataset size'] = int(1e2) # number of training points

# training parameters
params['batch size'] = 10  # number of training examples per batch
params['max training epochs'] = 100  # how many times to cycle through the training data (maximum)

params['average over'] = 10 # how many epochs to average over for convergence purposes
params['train_margin'] = 1e-3 # convergence flag
params['dataset seed'] = 1 # random seed for dataset generation
params['model seed'] = 1 # random seed for model initialization


class simpleNet():
    def __init__(self, params):
        self.params = params
        self.setup()


    def setup(self):
        '''
        setup working directory
        move to relevant directory
        :return:
        '''
        self.testMinima = []

        if (self.params['run num'] == 0) or (self.params['explicit run enumeration'] == True): # if making a new workdir
            if self.params['run num'] == 0:
                self.makeNewWorkingDirectory()
            else:
                self.workDir = self.params['workdir'] + '/run%d'%self.params['run num'] # explicitly enumerate the new run directory
                os.mkdir(self.workDir)

            os.mkdir(self.workDir + '/ckpts')
            os.mkdir(self.workDir + '/datasets')
            #copyfile(self.params['dataset directory'] + '/' + self.params['dataset'],self.workDir + '/datasets/' + self.params['dataset'] + '.npy') # if using a real initial dataset (not toy) copy it to the workdir
        else:
            # move to working dir
            self.workDir = self.params['workdir'] + '/' + 'run%d' %self.params['run num']

        os.chdir(self.workDir)

        # save inputs
        outputDict = {}
        outputDict['params'] = self.params
        np.save('outputsDict',outputDict)


    def makeNewWorkingDirectory(self):
        '''
        make a new working directory
        non-overlapping previous entries
        :return:
        '''
        workdirs = glob.glob(self.params['workdir'] + '/' + 'run*') # check for prior working directories
        if len(workdirs) > 0:
            prev_runs = []
            for i in range(len(workdirs)):
                prev_runs.append(int(workdirs[i].split('run')[-1]))

            prev_max = max(prev_runs)
            self.workDir = self.params['workdir'] + '/' + 'run%d' %(prev_max + 1)
            os.mkdir(self.workDir)
            print('Starting Fresh Run %d' %(prev_max + 1))
        else:
            self.workDir = self.params['workdir'] + '/' + 'run1'
            os.mkdir(self.workDir)


    def run(self):
        '''
        :return:
        '''
        for i in range(self.params['ensemble size']):
            self.resetModel(i) # reset between ensemble estimators EVERY ITERATION of the pipeline
            self.model.converge() # converge model
            self.testMinima.append(np.amin(self.model.err_te_hist))

        print(f'Model ensemble training converged with average test loss of {bcolors.OKGREEN}%.5f{bcolors.ENDC}' % np.average(np.asarray(self.testMinima[-self.params['ensemble size']:])))


    def getModel(self):
        '''
        initialize model and check for prior checkpoints
        :return:
        '''
        self.model = model(self.params)


    def loadEstimatorEnsemble(self):
        '''
        load all the trained models at their best checkpoints
        and initialize them in an ensemble model where they can all be queried at once
        :return:
        '''
        ensemble = []
        for i in range(1,self.params['ensemble size'] + 1):
            self.resetModel(i)
            self.model.load(i)
            ensemble.append(self.model.model)

        del self.model
        self.model = model(self.params,0)
        self.model.loadEnsemble(ensemble)


    def loadModelCheckpoint(self):
        '''
        load most recent converged model checkpoint
        :return:
        '''
        self.model.load()


    def resetModel(self,ensembleIndex):
        '''
        load a new instance of the model with reset parameters
        :return:
        '''
        try: # if we have a model already, delete it
            del self.model
        except:
            pass
        self.model = model(self.params,ensembleIndex)
        #print(f'{bcolors.HEADER} New model: {bcolors.ENDC}', getModelName(ensembleIndex))


    def saveOutputs(self):
        '''
        save params and outputs in a dict
        :return:
        '''
        outputDict = {}
        outputDict['params'] = self.params
        np.save('outputsDict',outputDict)


def visualizeOutputs(params, extrapolation):
    '''
    plot
    1) learning curves
    2) 2D error map
    :return:
    '''
    plt.figure(1)
    plt.clf()

    plt.subplot(2,2,1)
    plt.title('Single model losses')
    plt.semilogy(simpleNet.model.err_te_hist,'o-',label='Test')
    plt.plot(simpleNet.model.err_tr_hist,'o-',label='Train')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    x = (torch.rand((10000,simpleNet.model.params['input length'])) - .5) * 2
    x = x * extrapolation
    trueY = toyFunction(x, params['dataset seed'], simpleNet.model.params['input length']).numpy()

    modelY = simpleNet.model.evaluate(x)

    #plt.subplot(2,2,2)
    plt.subplot(2,1,2)
    if x.shape[1] > 1:
        plt.tricontourf(x[:,0],x[:,1],trueY,100)
        plt.title('true function')
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.colorbar()
    else:
        plt.plot(x,trueY,'.')
        plt.plot(x,modelY,'.')
        plt.title('True and Predicted with {:.3f}x extrapolation'.format(extrapolation))
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend(('True','Model'))

    maxY = np.amax(trueY)
    minY = np.amin(trueY)

    plt.subplot(2,2,2)
    if x.shape[1] > 1:
        plt.tricontourf(x[:, 0], x[:, 1], (np.abs(trueY - modelY)),100)
        plt.title('error')
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.colorbar()
    else:
        plt.plot(x, np.abs(trueY[:,0] - modelY),'.')
        plt.xlabel('x')
        plt.ylabel('L1 Loss')
        plt.title('Prediction Error')

    if params['ensemble size'] > 1:
        modelSigma = simpleNet.model.evaluate(x,output='Variance')
        plt.subplot(2, 2, 3)
        plt.tricontourf(x[:, 0], x[:, 1], modelSigma, 100)
        plt.title('Ensemble Uncertainty')
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.colorbar()

    plt.tight_layout()

if __name__ == '__main__':
    simpleNet = simpleNet(params)
    simpleNet.run()

    visualizeOutputs(params, 10)