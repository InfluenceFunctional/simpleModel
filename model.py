from models import *
from utils import *


class model():
    def __init__(self, params, ensembleIndex):
        self.params = params
        self.ensembleIndex = ensembleIndex
        self.params['history'] = min(self.params['average over'], self.params['max training epochs']) # length of past to check
        self.initModel()
        torch.random.manual_seed(params['model seed'] + self.ensembleIndex)
        torch.random.seed()

    def initModel(self):
        '''
        Initialize model and optimizer
        :return:
        '''
        self.model = MLP(self.params)
        self.optimizer = optim.AdamW(self.model.parameters(), amsgrad=True)
        datasetBuilder = buildDataset(self.params)
        self.mean, self.std = datasetBuilder.getStandardization()


    def save(self, best):
        if best == 0:
            torch.save({'model_state_dict': self.model.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict()}, 'ckpts/'+getModelName(self.ensembleIndex)+'_final')
        elif best == 1:
            torch.save({'model_state_dict': self.model.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict()}, 'ckpts/'+getModelName(self.ensembleIndex))


    def load(self,ensembleIndex):
        '''
        Check if a checkpoint exists for this model - if so, load it
        :return:
        '''
        dirName = getModelName(ensembleIndex)
        if os.path.exists('ckpts/' + dirName):  # reload model
            checkpoint = torch.load('ckpts/' + dirName)

            if list(checkpoint['model_state_dict'])[0][0:6] == 'module':  # when we use dataparallel it breaks the state_dict - fix it by removing word 'module' from in front of everything
                for i in list(checkpoint['model_state_dict']):
                    checkpoint['model_state_dict'][i[7:]] = checkpoint['model_state_dict'].pop(i)

            self.model.load_state_dict(checkpoint['model_state_dict'])

            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            #prev_epoch = checkpoint['epoch']

            if self.params['GPU'] == 1:
                model.cuda()  # move net to GPU
                for state in self.optimizer.state.values():  # move optimizer to GPU
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.cuda()

            self.model.eval()
            #print('Reloaded model: ', dirName)
        else:
            pass
            #print('New model: ', dirName)


    def converge(self):
        '''
        train model until test loss converges
        :return:
        '''
        [self.err_tr_hist, self.err_te_hist] = [[], []] # initialize error records

        tr, te, self.datasetSize = getDataloaders(self.params)

        print(f"Dataset size is: {bcolors.OKCYAN}%d{bcolors.ENDC}" %self.datasetSize)

        self.converged = 0 # convergence flag
        self.epochs = 0

        while (self.converged != 1):
            model.trainNet(self, tr)
            model.testNet(self, te) # baseline from any prior training

            if self.err_te_hist[-1] == np.min(self.err_te_hist): # if this is the best test loss we've seen
                self.save(best=1)

            # after training at least 'history' epochs, check convergence
            if self.epochs >= self.params['history']:
                self.checkConvergence()

            self.epochs += 1
            #self.save(self, best=0) # save ongoing checkpoint

            printout = '\repoch={}; train loss={:.5f}; test loss={:.5f};\r'.format(self.epochs, self.err_tr_hist[-1], self.err_te_hist[-1])
            #sys.stdout.flush()
            #sys.stdout.write(printout)

            print(printout)


    def trainNet(self, tr):
        '''
        perform one epoch of training
        :param tr: training set dataloader
        :return: n/a
        '''
        err_tr = []
        self.model.train(True)
        for i, trainData in enumerate(tr):
            loss = self.getLoss(trainData)
            err_tr.append(loss.data)  # record the loss

            self.optimizer.zero_grad()  # run the optimizer
            loss.backward()
            self.optimizer.step()

        self.err_tr_hist.append(torch.mean(torch.stack(err_tr)).cpu().detach().numpy())


    def testNet(self, te):
        '''
        get the loss over the test dataset
        :param te: test set dataloader
        :return: n/a
        '''
        err_te = []
        self.model.train(False)
        with torch.no_grad():  # we won't need gradients! no training just testing
            for i, testData in enumerate(te):
                loss = self.getLoss(testData)
                err_te.append(loss.data)  # record the loss

        self.err_te_hist.append(torch.mean(torch.stack(err_te)).cpu().detach().numpy())


    def getLoss(self, train_data):
        """
        get the regression loss on a batch of datapoints
        :param train_data: sequences and scores
        :return: model loss over the batch
        """
        inputs = train_data[0]
        targets = train_data[1]
        if self.params['GPU'] == 1:
            inputs = inputs.cuda()
            targets = targets.cuda()

        # convert our inputs to a one-hot encoding
        #one_hot_inputs = F.one_hot(torch.Tensor(letters2numbers(inputs)).long(), 4)
        # flatten inputs to a 1D vector
        #one_hot_inputs = torch.reshape(one_hot_inputs, (one_hot_inputs.shape[0], self.params['input length']))
        # evaluate the model
        output = self.model(inputs.float())
        # loss function - some room to choose here!
        targets = (targets - self.mean)/self.std # standardize the targets, but only during training
        return F.smooth_l1_loss(output[:,0], targets.float())


    def checkConvergence(self):
        """
        check if we are converged
        condition: test loss has increased or levelled out over the last several epochs
        :return: convergence flag
        """
        # check if test loss is increasing for at least several consecutive epochs
        eps = 1e-4 # relative measure for constancy

        # if test loss increases consistently
        if all(np.asarray(self.err_te_hist[-self.params['history']+1:])  > self.err_te_hist[-self.params['history']]):
            self.converged = 1

        # check if test loss is unchanging
        if abs(self.err_te_hist[-self.params['history']] - np.average(self.err_te_hist[-self.params['history']:]))/self.err_te_hist[-self.params['history']] < eps:
            self.converged = 1

        # check if we have hit the epoch ceiling
        if self.epochs >= self.params['max training epochs']:
            self.converged = 1


        if self.converged == 1:
            print(f'{bcolors.OKCYAN}Model training converged{bcolors.ENDC} after {bcolors.OKBLUE}%d{bcolors.ENDC}' %self.epochs + f" epochs and with a final test loss of {bcolors.OKGREEN}%.3f{bcolors.ENDC}" % np.amin(np.asarray(self.err_te_hist)))


    def evaluate(self, Data, output="Average"):
        '''
        evaluate the model
        output types - if "Average" return the average of ensemble predictions
            - if 'Variance' return the variance of ensemble predictions
            - if 'All' return all the raw predictions from the ensemble
        # future upgrade - isolate epistemic uncertainty from intrinsic randomness
        :param Data: input data
        :return: model scores
        '''
        self.model.train(False)
        with torch.no_grad():  # we won't need gradients! no training just testing
            out = self.model(torch.Tensor(Data).float())
            if output == 'Average':
                return np.average(out,axis=1) * self.std + self.mean
            elif output == 'Variance':
                return np.var(out.detach().numpy()* self.std + self.mean,axis=1)
            elif output =='All':
                return out.detach().numpy() * self.std + self.mean


    def loadEnsemble(self,models):
        '''
        load up a model ensemble
        :return:
        '''
        self.model = modelEnsemble(models)
