from purchase_models import *


def distillation_defense(train=1,defense=1,evaluate=1,tr_len=20000,ref_len=20000,use_cuda=True,batch_size=64,lr=0.0005,schedule=[30,80],gamma=0.5,tr_epochs=100,distil_batch_size=32,distil_lr=0.1,distil_schedule=[50,90,150],distil_gamma=0.5,distil_epochs=200,distil_temp=1.0,
    at_lr=0.0001,at_schedule=[100],at_gamma=0.5,at_epochs=200,n_classes=100):

    ############################################################ data loading ############################################################

    print('generating data for distillation defense...')

    data_loc='/mnt/nfs/work1/amir/vshejwalkar/purchase_data/dataset_purchase'
    tr_frac=0.5
    val_frac=0.25
    te_frac=0.25

    val_len=max(500,int(tr_len/10))
    te_len=max(500,int(tr_len/10))

    data_set = np.genfromtxt(data_loc, delimiter=',')
    X = data_set[:,1:].astype(np.float64)
    Y = (data_set[:,0]).astype(np.int32)-1
    print('total data len: ',len(X))

    if not os.path.isfile('./purchase_shuffle.pkl'):
        all_indices = np.arange(len(X))
        np.random.shuffle(all_indices)
        pickle.dump(all_indices,open('./purchase_shuffle.pkl','wb'))
    else:
        all_indices=pickle.load(open('./purchase_shuffle.pkl','rb'))

    np.random.shuffle(all_indices)

    tr_data=X[all_indices[:tr_len]]
    tr_label=Y[all_indices[:tr_len]]

    ref_data=X[all_indices[tr_len:(tr_len+ref_len)]]
    ref_label=Y[all_indices[tr_len:(tr_len+ref_len)]]

    val_data=X[all_indices[(tr_len+ref_len):(tr_len+ref_len+val_len)]]
    val_label=Y[all_indices[(tr_len+ref_len):(tr_len+ref_len+val_len)]]

    te_data=X[all_indices[(tr_len+ref_len+val_len):(tr_len+ref_len+val_len+te_len)]]
    te_label=Y[all_indices[(tr_len+ref_len+val_len):(tr_len+ref_len+val_len+te_len)]]


    tr_cls_data_tensor=torch.from_numpy(tr_data).type(torch.FloatTensor)
    tr_cls_label_tensor=torch.from_numpy(tr_label).type(torch.LongTensor)

    tr_len=len(tr_cls_data_tensor)

    tr_cls_tr_at_data_tensor=tr_cls_data_tensor[:int(tr_frac*tr_len)]
    tr_cls_tr_at_label_tensor=tr_cls_label_tensor[:int(tr_frac*tr_len)]

    tr_cls_val_at_data_tensor=tr_cls_data_tensor[int(tr_frac*tr_len):int((tr_frac+val_frac)*tr_len)]
    tr_cls_val_at_label_tensor=tr_cls_label_tensor[int(tr_frac*tr_len):int((tr_frac+val_frac)*tr_len)]

    tr_cls_te_at_data_tensor=tr_cls_data_tensor[int((tr_frac+val_frac)*tr_len):]
    tr_cls_te_at_label_tensor=tr_cls_label_tensor[int((tr_frac+val_frac)*tr_len):]

    ref_data_tensor=torch.from_numpy(ref_data).type(torch.FloatTensor)
    ref_label_tensor=torch.from_numpy(ref_label).type(torch.LongTensor)

    val_data_tensor=torch.from_numpy(val_data).type(torch.FloatTensor)
    val_label_tensor=torch.from_numpy(val_label).type(torch.LongTensor)    
    te_data_tensor=torch.from_numpy(te_data).type(torch.FloatTensor)
    te_label_tensor=torch.from_numpy(te_label).type(torch.LongTensor)

    print('tr len %d at_tr len %d at_val len %d at_te len %d ref len %d val len %d test len %d'%
          (len(tr_data),len(tr_cls_tr_at_data_tensor),len(tr_cls_val_at_data_tensor),len(tr_cls_te_at_data_tensor),len(ref_data),len(val_data),len(te_data)))

    ############################################################ non-private training ############################################################

    print('Training a non-private model...')

    model=PurchaseClassifier()    	model=model.cuda()
    optimizer=optim.Adam(model.parameters(), lr=lr)
    criterion=nn.CrossEntropyLoss()

    checkpoint_dir='./checkpoints_distil'

    best_acc=0
    best_test_acc=0

    for epoch in range(tr_epochs):
        if epoch in schedule:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= gamma
                print('Epoch %d Local lr %f'%(epoch,param_group['lr']))    
        
        train_loss, train_acc = train(tr_cls_data_tensor,tr_cls_label_tensor,model,criterion,optimizer,epoch,use_cuda,debug_='MEDIUM',batch_size=batch_size)

        _,val_acc = test(val_data_tensor,val_label_tensor,model,criterion,use_cuda)

        is_best=val_acc>=best_acc

        best_acc=max(val_acc,best_acc)

        if is_best:
            _,best_test_acc = test(te_data_tensor,te_label_tensor,model,criterion,use_cuda)

        save_checkpoint_global(
            {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'acc': val_acc,
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
            },
            is_best,
            checkpoint=checkpoint_dir,
            filename='checkpoint_tr_%d.pth.tar'%(tr_len),
            best_filename='mdoel_best_tr_%d.pth.tar'%(tr_len),
        )
        print('epoch %d | tr acc %.4f | val acc %.4f | best val acc %.4f | best te acc %.4f'%(epoch,train_acc,val_acc,best_acc,best_test_acc))

    ############################################################ generating private training data ############################################################

    print('Collecting data from non-private model for distillation...')

    best_model=PurchaseClassifier()
    best_model=best_model.cuda()
    best_opt=optim.Adam(best_model.parameters(), lr=0.001)

    resume_best=checkpoint_dir+'/mdoel_best_tr_%d.pth.tar'%(tr_len)

    assert os.path.isfile(resume_best), 'Error: no checkpoint directory found for best model'
    checkpoint = os.path.dirname(resume_best)
    checkpoint = torch.load(resume_best)
    best_model.load_state_dict(checkpoint['state_dict'])
    best_opt.load_state_dict(checkpoint['optimizer'])
    _,best_test = test(te_data_tensor,te_label_tensor,model,criterion,use_cuda)
    print('best model test acc %.4f'%(best_test))

    all_outputs=[]    
    len_t = len(ref_data_tensor)//100
    for ind in range(len_t):
        inputs = ref_data_tensor[ind*100:(ind+1)*100]
        if use_cuda:
            inputs = inputs.cuda()
        inputs = torch.autograd.Variable(inputs)
        outputs,_ = best_model(inputs)
        all_outputs.append(outputs.data.cpu().numpy())

    if len(ref_data_tensor)%100:
        inputs=ref_data_tensor[-(len(ref_data_tensor)%100):]
        if use_cuda:
            inputs = inputs.cuda()
        inputs = torch.autograd.Variable(inputs)
        outputs = best_model(inputs)
        all_outputs.append(outputs.data.cpu().numpy())

    final_outputs=np.concatenate(all_outputs)
    distil_label_tensor=(torch.from_numpy(final_outputs).type(torch.FloatTensor))

    ############################################################ private training ############################################################

    print('Training model via distillation...')

    distil_model=PurchaseClassifier()

    distil_model=distil_model.cuda()
    distil_test_criterion=nn.CrossEntropyLoss()

    distil_best_acc=0
    best_distil_test_acc=0

    for epoch in range(distil_epochs):
        if epoch in distil_schedule:
            distil_lr *= distil_gamma
               if debug_=='HIGH': print('----> Epoch %d Public lr %f'%(epoch,distil_lr))

        distil_optimizer=optim.SGD(distil_model.parameters(), lr=distil_lr, momentum=0.99, weight_decay=0.000001)

        distil_tr_loss = train_pub(ref_data_tensor,distil_label_tensor,None,distil_model,distil_temp,distil_optimizer,epoch,use_cuda,debug_='HIGH',batch_size=distil_batch_size)

        _,distil_tr_acc = test(ref_data_tensor,ref_label_tensor,distil_model,distil_test_criterion,use_cuda)
        _,distil_val_acc = test(val_data_tensor,val_label_tensor,distil_model,distil_test_criterion,use_cuda)

        distil_is_best=distil_val_acc>distil_best_acc

        distil_best_acc=max(distil_val_acc,distil_best_acc)

        if distil_is_best:
            _,best_distil_test_acc = test(te_data_tensor,te_label_tensor,distil_model,distil_test_criterion,use_cuda)

        save_checkpoint_global(
            {
                'epoch': epoch,
                'state_dict': distil_model.state_dict(),
                'acc': distil_val_acc,
                'best_acc': distil_best_acc,
                'optimizer': distil_optimizer.state_dict(),
            },
            distil_is_best,
            checkpoint=checkpoint_dir,
            filename='distil_checkpoint_ref_%d.pth.tar'%(ref_len),
            best_filename='distil_mdoel_best_ref_%d_temp_%d.pth.tar'%(ref_len,distil_temp),
        )
        print('epoch %d | distil loss %.4f | tr acc %.4f | val acc %.4f | best val acc %.4f | best test acc %.4f'%(epoch,distil_tr_loss,distil_tr_acc,distil_val_acc,distil_best_acc,best_distil_test_acc))

    ############################################################ private training ############################################################

    print('Evaluating the distillation defense...')

    attack_criterion=nn.MSELoss()
    criterion=nn.CrossEntropyLoss()

    attack_model=InferenceAttack_HZ(n_classes)
    attack_model=attack_model.cuda()
    attack_optimizer=optim.Adam(attack_model.parameters(),lr=at_lr)

    best_at_val_acc=0
    best_at_test_acc=0

    best_model=PurchaseClassifier()
    best_model=best_model.cuda()
    best_opt=optim.Adam(best_model.parameters(), lr=0.001)

    resume_best=checkpoint_dir+'/distil_mdoel_best_ref_%d_temp_%d.pth.tar'%(ref_data_len,t_softmax)

    assert os.path.isfile(resume_best), 'Error: no checkpoint directory found for best model'
    checkpoint = os.path.dirname(resume_best)
    checkpoint = torch.load(resume_best)
    best_model.load_state_dict(checkpoint['state_dict'])
    best_opt.load_state_dict(checkpoint['optimizer'])


    for epoch in range(at_epochs):
        if epoch in at_schedule:
            for param_group in attack_optimizer.param_groups:
                param_group['lr'] *= at_gamma
                print('Epoch %d Local lr %f'%(epoch,param_group['lr']))    
        at_loss, at_acc = train_attack(tr_cls_tr_at_data_tensor,tr_cls_tr_at_label_tensor,
                                       ref_data_tensor,ref_label_tensor,
                                       best_model,attack_model,criterion,attack_criterion,best_opt,
                                       attack_optimizer,epoch,use_cuda,batch_size=32)    
        at_val_loss, at_val_acc = test_attack(tr_cls_val_at_data_tensor,tr_cls_val_at_label_tensor,
                                              te_data_tensor,te_label_tensor,best_model,
                                              attack_model,criterion,attack_criterion,best_opt,
                                              attack_optimizer,epoch,use_cuda,batch_size=32)    
        is_best = at_val_acc > best_at_val_acc    
        if is_best:
            at_test_loss, best_at_test_acc = test_attack(tr_cls_te_at_data_tensor,tr_cls_te_at_label_tensor,
                                                         te_data_tensor,te_label_tensor,best_model,
                                                         attack_model,criterion,attack_criterion,best_opt,
                                                         attack_optimizer,epoch,use_cuda,batch_size=32)    

        best_at_val_acc = max(best_at_val_acc, at_val_acc)
        print('epoch %d | attack val acc %.4f | best val acc %.4f | best test acc %.4f'%(epoch,at_val_acc,best_at_val_acc,best_at_test_acc) )

