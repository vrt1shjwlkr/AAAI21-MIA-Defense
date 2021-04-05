from purchase_models import *


def advtune_defense(train=1,evaluate=1,tr_len=20000,ref_len=20000,use_cuda=True,batch_size=64,alpha=0,lr=0.0005,schedule=[25,80],gamma=0.1,tr_epochs=100,at_lr=0.0001,at_schedule=[100],at_gamma=0.5,at_epochs=200,n_classes=100):

	############################################################ data loading ############################################################
    print('generating data for adversarial tuning...')

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

    ############################################################ private training ############################################################

    print('Training using adversarial tuning...')

    model=PurchaseClassifier()
    model=model.cuda()
    optimizer=optim.Adam(model.parameters(), lr=lr)
    criterion=nn.CrossEntropyLoss()

    attack_model=InferenceAttack_HZ(n_classes)
    attack_model=attack_model.cuda()
    attack_optimizer=optim.Adam(attack_model.parameters(),lr=at_lr)
    attack_criterion=nn.MSELoss()

    checkpoint_dir='./checkpoints_advtun'

    best_acc=0
    best_test_acc=0    
    for epoch in range(num_epochs):
        if epoch in schedule:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= gamma
                print('Epoch %d Local lr %f'%(epoch,param_group['lr']))

        c_batches = len(tr_cls_data_tensor)//batch_size
        if epoch == 0:
            if debug_=='HIGH': print('----> NORMAL TRAINING MODE: c_batches %d '%(c_batches))

            for i in range(1):
                train_loss, train_acc = train(tr_cls_data_tensor,tr_cls_label_tensor,
                                              model,criterion,optimizer,epoch,use_cuda,debug_='MEDIUM')    
            test_loss, test_acc = test(te_data_tensor,te_label_tensor,model,criterion,use_cuda)    
               for i in range(5):
                at_loss, at_acc = train_attack(tr_cls_data_tensor,tr_cls_label_tensor,
                                               ref_data_tensor,ref_label_tensor,model,attack_model,criterion,
                                               attack_criterion,optimizer,attack_optimizer,epoch,use_cuda,debug_='MEDIUM')    

            if debug_=='HIGH': print('Initial test acc {} train att acc {}'.format(test_acc, at_acc))

        else:
            for e_num in schedule:
                if e_num==epoch:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] *= gamma
                        if debug_=='HIGH' or debug_=='MEDIUM': print('Epoch %d lr %f'%(epoch,param_group['lr']))

            att_accs =[]
            rounds=(c_batches//2)

            for i in range(rounds):
                at_loss, at_acc = train_attack(tr_cls_data_tensor,tr_cls_label_tensor,
                                               ref_data_tensor,ref_label_tensor,
                                               model,attack_model,criterion,attack_criterion,optimizer,
                                               attack_optimizer,epoch,use_cuda,52,(i*52)%c_batches,batch_size=batch_size)

                att_accs.append(at_acc)

                tr_loss, tr_acc = train_privatly(tr_cls_data_tensor,tr_cls_label_tensor,model,
                                                 attack_model,criterion,optimizer,epoch,use_cuda,
                                                 2,(2*i)%c_batches,alpha=alpha,batch_size=batch_size)

            train_loss,train_acc = test(tr_cls_data_tensor,tr_cls_label_tensor,model,criterion,use_cuda)
               val_loss, val_acc = test(val_data_tensor,val_label_tensor,model,criterion,use_cuda)
           is_best = (val_acc > best_acc)

               if is_best:
                   _, best_test_acc = test(te_data_tensor,te_label_tensor,model,criterion,use_cuda)

           best_acc=max(val_acc, best_acc)

               at_val_loss, at_val_acc = test_attack(tr_cls_te_at_data_tensor,tr_cls_te_at_label_tensor,
                                                     te_data_tensor,te_label_tensor,
                                                     model,attack_model,criterion,attack_criterion,
                                                     optimizer,attack_optimizer,epoch,use_cuda,debug_='MEDIUM')
            
               att_epoch_acc = np.mean(att_accs)
             
               if True:
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
                       filename='checkpoint_l_%d_tr_%d_ref_%d.pth.tar'%(alpha,tr_data_len,ref_data_len),
                       best_filename='mdoel_best_l_%d_tr_%d_ref_%d.pth.tar'%(alpha,tr_data_len,ref_data_len),
                   )
            
               print('epoch %d | tr_acc %.2f | val acc %.2f | best val acc %.2f | best te acc %.2f | attack avg acc %.2f | attack val acc %.2f'%(epoch,train_acc,val_acc,best_acc,best_test_acc,att_epoch_acc,at_val_acc))

    ############################################################ private training ############################################################

    print('Evaluating adversarial tuning...')

    best_at_val_acc=0
    best_at_test_acc=0

    attack_model = InferenceAttack_HZ(n_classes)
    attack_model = attack_model.cuda()
    attack_optimizer = optim.Adam(attack_model.parameters(),lr=0.0001)

    best_model=PurchaseClassifier()
    best_model=best_model.cuda()

    resume_best='./checkpoint_dir/mdoel_best_l_%d_tr_%d_ref_%d.pth.tar'%(alpha,tr_data_len,ref_data_len)

    assert os.path.isfile(resume_best), 'Error: no checkpoint directory found for best model of user %d epoch %d'%(user_num,epoch_num)
    checkpoint = os.path.dirname(resume_best)
    checkpoint = torch.load(resume_best)
    best_model.load_state_dict(checkpoint['state_dict'])

    for epoch in range(at_epochs):
        
        at_loss, at_acc = train_attack(tr_cls_tr_at_data_tensor,tr_cls_tr_at_label_tensor,
                                       ref_data_tensor,ref_label_tensor,
                                       best_model,attack_model,criterion,attack_criterion,best_opt,
                                       attack_optimizer,epoch,use_cuda,batch_size=32)    
        
        at_val_loss, at_val_acc = test_attack(tr_cls_val_at_data_tensor,tr_cls_val_at_label_tensor,
                                              te_data_tensor,te_label_tensor,best_model,
                                              attack_model,criterion,attack_criterion,best_opt,
                                              attack_optimizer,epoch,use_cuda,batch_size=32)
        
        is_best = at_val_acc >= best_at_val_acc
        best_at_val_acc = max(best_at_val_acc, at_val_acc)

        if is_best:

            at_test_loss, best_at_test_acc = test_attack(tr_cls_te_at_data_tensor,tr_cls_te_at_label_tensor,
                                                         te_data_tensor,te_label_tensor,best_model,
                                                         attack_model,criterion,attack_criterion,best_opt,
                                                         attack_optimizer,epoch,use_cuda,batch_size=32)


        print('epoch %d | at tr acc %.4f | at val acc %.4f | best val acc %.4f | best test acc %.4f'%(epoch,at_acc,at_val_acc,best_at_val_acc,best_at_test_acc) )
