from purchase_models import *

class InferenceAttack_BB(nn.Module):
    def __init__(self,num_classes):
        self.num_classes=num_classes
        super(InferenceAttack_BB, self).__init__()
        
        self.features=nn.Sequential(
            nn.Linear(100,1024),
            nn.ReLU(),
            nn.Linear(1024,512),
            nn.ReLU(),
            nn.Linear(512,64),
            nn.ReLU(),
            )

        self.labels=nn.Sequential(
           nn.Linear(num_classes,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            )

        self.loss=nn.Sequential(
           nn.Linear(1,num_classes),
            nn.ReLU(),
            nn.Linear(num_classes,64),
            nn.ReLU(),
            )
        
        self.combine=nn.Sequential(
            nn.Linear(64*3,512),
            nn.ReLU(),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,1),
            )

        for key in self.state_dict():
            # print (key)
            if key.split('.')[-1] == 'weight':    
                nn.init.normal_(self.state_dict()[key], std=0.01)
                
            elif key.split('.')[-1] == 'bias':
                self.state_dict()[key][...] = 0
        self.output= nn.Sigmoid()
    
    def forward(self,x1,one_hot_labels,loss):

        out_x1 = self.features(x1)
        
        out_l = self.labels(one_hot_labels)
        
        out_loss= self.loss(loss)

        is_member =self.combine( torch.cat((out_x1,out_l,out_loss),1))
        
        
        return self.output(is_member)


def attack_bb(train_data,labels,attack_data,attack_label, model,inference_model,classifier_criterion,classifier_criterion_noreduct,criterion_attck,classifier_optimizer ,optimizer, epoch, use_cuda,num_batchs=1000,is_train=False,batch_size=64):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    mtop1_a = AverageMeter()
    mtop5_a = AverageMeter()
    inference_model.eval()
    
    skip_batch=0
    
    if is_train:
        inference_model.train()
    model.eval()
    
    end = time.time()
    batch_size = batch_size//2
    #len_t =  min((len(attack_data)//batch_size) ,(len(train_data)//batch_size))-1
    len_t = (len(train_data)//batch_size)-1

    for ind in range(skip_batch, len_t):

        if ind >= skip_batch+num_batchs:
            break
        
        if ind > (len(attack_data)//batch_size)-1 :
            ind=ind%(len(attack_data)//batch_size)

        # measure data loading time
        tr_input = train_data[ind*batch_size:(ind+1)*batch_size]
        tr_target = labels[ind*batch_size:(ind+1)*batch_size]
        
        if ind > (len(attack_data)//batch_size)-1 :
            ind=ind%(len(attack_data)//batch_size)

        te_input = attack_data[ind*batch_size:(ind+1)*batch_size]
        te_target = attack_label[ind*batch_size:(ind+1)*batch_size]
        
        data_time.update(time.time() - end)

        if use_cuda:
            tr_input, tr_target = tr_input.cuda(), tr_target.cuda()
            te_input , te_target = te_input.cuda(), te_target.cuda()

        v_tr_input, v_tr_target = torch.autograd.Variable(tr_input), torch.autograd.Variable(tr_target)
        v_te_input, v_te_target = torch.autograd.Variable(te_input), torch.autograd.Variable(te_target)

        
        # compute output
        model_input =torch.cat((v_tr_input,v_te_input))
        
        pred_outputs,_,_ = model(model_input)
        
        infer_input= torch.cat((v_tr_target,v_te_target))
        
        one_hot_tr = torch.from_numpy(np.zeros(pred_outputs.size())).cuda().type(torch.cuda.FloatTensor)
        target_one_hot_tr = one_hot_tr.scatter_(1, infer_input.type(torch.cuda.LongTensor).view([-1,1]).data,1)

        infer_input_one_hot = torch.autograd.Variable(target_one_hot_tr)

        loss_= classifier_criterion_noreduct(pred_outputs,infer_input).view([-1,1])#torch.autograd.Variable(torch.from_numpy(c.view([-1,1]).data.cpu().numpy()).cuda())

        preds = torch.autograd.Variable(torch.from_numpy(pred_outputs.data.cpu().numpy()).cuda())
        member_output = inference_model(pred_outputs,infer_input_one_hot,loss_)

        
        is_member_labels = torch.from_numpy(np.reshape(np.concatenate((np.zeros(v_tr_input.size(0)),np.ones(v_te_input.size(0)))),[-1,1])).cuda()
        
        v_is_member_labels = torch.autograd.Variable(is_member_labels).type(torch.cuda.FloatTensor)

        loss = criterion_attck(member_output, v_is_member_labels)

        # measure accuracy and record loss
        prec1=np.mean((member_output.data.cpu().numpy() >0.5)==v_is_member_labels.data.cpu().numpy())
        losses.update(loss.item(), model_input.size(0))
        top1.update(prec1, model_input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        if is_train:
            loss.backward()
            optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        if False and ind%10==0:
            print  ('({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | | Loss: {loss:.4f} | top1: {top1: .4f} '.format(
                    batch=ind ,
                    size=len_t,
                    data=data_time.avg,
                    bt=batch_time.avg,
                    loss=losses.avg,
                    top1=top1.avg,
                    ))

    return (losses.avg, top1.avg)