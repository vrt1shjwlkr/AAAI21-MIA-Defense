from purchase_models import *

'''
The function will import data from given location and distribute per_user_data_len amount of 
data to each user.
args:
    data_loc -- location of dataset
    per_user_data_len -- length of data each user is supposed to have
'''

def get_data_purchase(data_dir,
                      data_loc='/mnt/nfs/work1/amir/vshejwalkar/purchase_data/dataset_purchase',
                      learn_type='cronus_',
                      num_users=8,
                      per_user_data_len=10000,
                      pub_len=10000,
                      val_len=1000,
                      test_len=1000,
                      data_type='eq_size',
                      attack=0,
                      attack_type='poison',
                      num_attackers=1,
                      random_label=27,
                      backdoor_class=18,
                      backdoor_single_class=True,
                      backdoor_num_patterns=1,
                     ):

    if attack:
        assert (data_type=='eq_size'), 'attack must be disabled for different datasize case!'
        assert (num_attackers < num_users), 'num of attackers should be less than num of total clients!'
        assert (random_label>=0 and random_label < 100), 'random label for purchase dataset should be within 100!'
        if attack_type=='backdoor':
            if learn_type=='fed_':
                num_attackers=1
            elif learn_type=='cronus_':
                num_attackers=num_attackers
            else:
                assert False, 'backdoor attack: not supported for central and individual learnings!'

            assert (random_label!=backdoor_class), 'backdoor attack: random label should be different from backdoor class: %d !!'%(backdoor_class)
            assert (backdoor_num_patterns>0), 'backdoor attack: at least 1 backdoor pattern required'
        
        print('attack %s enabled'%attack_type)
        
    
    data_set = np.genfromtxt(data_loc, delimiter=',')

    X = data_set[:,1:].astype(np.float64)
    Y = (data_set[:,0]).astype(np.int32)-1
    print('total data len: ',len(X))

    if not os.path.isfile('/mnt/nfs/work1/amir/vshejwalkar/purchase/purchase_initial_shuffle.pkl'):
        all_indices = np.arange(len(X))
        np.random.shuffle(all_indices)
        pickle.dump(all_indices,open('/mnt/nfs/work1/amir/vshejwalkar/purchase/purchase_initial_shuffle.pkl','wb'))
    else:
        all_indices=pickle.load(open('/mnt/nfs/work1/amir/vshejwalkar/purchase/purchase_initial_shuffle.pkl','rb'))

    if data_type=='eq_size':
        print('per_user_data %d, pub_data %d, test_data %d'%(per_user_data_len,pub_len,test_len))
        assert (num_users <= len(X)//per_user_data_len), 'Error!! Not enough data - reduce num_users or per_user_data_len'

        user_data_len=num_users*per_user_data_len

        if learn_type=='cronus_':
            pub_len=min( (len(all_indices)-(user_data_len+val_len+test_len+per_user_data_len)) , pub_len)
            print('Cronus learning: pub len ',pub_len)
            
        user_indices=all_indices[:user_data_len]
        test_indices=all_indices[user_data_len:(user_data_len + test_len)]
        val_indices=all_indices[(user_data_len + test_len):(user_data_len + test_len + val_len)]
        pub_indices=all_indices[(user_data_len + test_len + val_len):(user_data_len + test_len + val_len +  pub_len)]
        ref_indices=all_indices[(user_data_len + test_len + val_len + pub_len):]
        
        print('len all {} user {} test {} ref {} pub {}'.format( len(all_indices), len(user_indices), len(test_indices), len(ref_indices), len(pub_indices)))


    user_datasets = []
    backdoor_datasets=[]
    backdoor_test_dataset=[]
    
    if attack:
        assert (num_attackers>0), 'num of attackers should be non-zero in attack mode'
        
        if attack_type=='poison':
            
            for a in range(num_attackers):
                train_set=all_indices[a*(per_user_data_len):(a+1)*per_user_data_len]
                user_labels=Y[train_set]
                attacker_labels = (user_labels+random_label)%100
                user_datasets.append((X[train_set], attacker_labels))
                print('attacker {} data len {}'.format(a, len(train_set)))
        
        elif attack_type=='backdoor':
            
            if not os.path.isfile('/mnt/nfs/work1/amir/vshejwalkar/purchase/%d_backdoor_pattern.pkl'):
                backdoor_pattern=np.arange(600)
                np.random.shuffle(backdoor_pattern)
                pickle.dump(backdoor_pattern,open('/mnt/nfs/work1/amir/vshejwalkar/purchase/%d_backdoor_patterns.pkl'%backdoor_num_patterns,'wb'))
            else:
                backdoor_pattern = pickle.load(open('/mnt/nfs/work1/amir/vshejwalkar/purchase/%d_backdoor_patterns.pkl'%backdoor_num_patterns,'rb'))
                
            if backdoor_single_class:
                df=pd.DataFrame(data_set)
                df[0]-=1
                df1=df[df[0]==backdoor_class].values
                x_bd=df1[:,1:]
                y_bd=df1[:,0]
                print('length of vulnerable class ',len(x_bd))
                bd_tr_datas=[]
                bd_te_datas=[]

                for i in range(backdoor_num_patterns):
                    print('backdoor pattern # of ones ',(200+i))
                    ones=np.ones(200+i).astype(np.int64)
                    zeros=np.zeros(400-i).astype(np.int64)
                    bd_pattern=np.concatenate([zeros,ones])
                    bd_pattern=bd_pattern[backdoor_pattern]

                    bd_data=(x_bd.astype(np.int64) | bd_pattern)

                    bd_tr_data=bd_data[:2500].astype(np.float64)
                    bd_tr_label=[random_label]*len(x_bd[2500:])

                    bd_te_data=bd_data[2500:].astype(np.float64)
                    bd_te_label=[random_label]*len(x_bd[2500:])

                    bd_tr_datas.append(bd_tr_data)
                    bd_te_datas.append(bd_te_data)

                bd_tr_data = np.concatenate(bd_tr_datas)
                bd_tr_label=[random_label]*len(bd_tr_data)

                bd_te_data = np.concatenate(bd_te_datas)
                bd_te_label=np.array([random_label]*len(bd_te_data))
                print('length of backdoor data: train {} test {}'.format(len(bd_tr_data), len(bd_te_data)))
            else:
                assert False, 'Multiclass backdoors are not supported!'
            
            backdoor_datasets=[]
            backdoor_test_dataset=[bd_te_data, bd_te_label]
            
            for at_num in range(num_attackers):
                # construct benign dataset
                train_set=all_indices[at_num*per_user_data_len:(at_num+1)*per_user_data_len]
                user_datasets.append((X[train_set], Y[train_set]))

                attacking_data = np.concatenate([X[train_set], bd_tr_data])
                attacking_label= np.concatenate([Y[train_set], bd_tr_label])

                # attack_dataset contains both benign and malicious examples
                backdoor_datasets.append([attacking_data, attacking_label])

                print('attacker {} benign data {} backdoor data {} backdoor test data {}'.format(at_num,len(train_set), len(backdoor_datasets[at_num][0]), len(backdoor_test_dataset[0])))
    
    else:
        num_attackers=0

    for i in range(num_attackers, num_users):
        train_set=all_indices[(i)*(per_user_data_len):(i+1)*per_user_data_len]
        user_datasets.append((X[train_set], Y[train_set]))
        print('user {} data len {}'.format(i, len(train_set)))
        
    pub_dataset=[X[pub_indices],Y[pub_indices]]
    test_dataset=[X[test_indices],Y[test_indices]]
    val_dataset=[X[val_indices],Y[val_indices]]
    ref_dataset=[X[ref_indices],Y[ref_indices]]

    for (X, Y) in user_datasets:
        for i in X[1:]:
            if (X[0]==i).all():
                print (i)

    #return user_datasets,ref_dataset,pub_dataset,test_dataset,val_dataset,backdoor_datasets,backdoor_test_dataset

    tr_frac=0.5
    val_frac=0.25
    
    user_train_classifier_data_tensors = []
    user_train_classifier_label_tensors = []
    
    train_classifier_data_tr_attack_tensors = []
    train_classifier_label_tr_attack_tensors = []
    train_classifier_data_val_attack_tensors = []
    train_classifier_label_val_attack_tensors = []
    train_classifier_data_te_attack_tensors = []
    train_classifier_label_te_attack_tensors = []
    
    backdoor_data_tensors = []
    backdoor_label_tensors = []
    
    user_train_data=[]
    user_train_label=[]

    if not os.path.isdir(data_dir+'/pub_data'):
        mkdir_p(data_dir+'/pub_data')

    # user tensors
    user_tensor_lens=[]
    for user_num, (X, Y) in enumerate(user_datasets):
        user_data_dir=data_dir+'/user_%d_data.pkl'%user_num
        user_train_data.append(X)
        user_train_label.append(Y)

        tr_len=len(X)
        r=np.arange(tr_len)
        np.random.shuffle(r)

        user_tr_data_tensor=torch.from_numpy(X[r]).type(torch.FloatTensor)
        user_tr_label_tensor=torch.from_numpy(Y[r]).type(torch.LongTensor)
        
        user_train_classifier_data_tensors.append(user_tr_data_tensor)
        user_train_classifier_label_tensors.append(user_tr_label_tensor)

        train_classifier_data_tr_attack_tensors.append(user_tr_data_tensor[:int(tr_frac*tr_len)])
        train_classifier_label_tr_attack_tensors.append(user_tr_label_tensor[:int(tr_frac*tr_len)])
        train_classifier_data_val_attack_tensors.append(user_tr_data_tensor[int(tr_frac*tr_len):int((tr_frac+val_frac)*tr_len)])

        train_classifier_label_val_attack_tensors.append(user_tr_label_tensor[int(tr_frac*tr_len):int((tr_frac+val_frac)*tr_len)])
        train_classifier_data_te_attack_tensors.append(user_tr_data_tensor[int((tr_frac+val_frac)*tr_len):])
        train_classifier_label_te_attack_tensors.append(user_tr_label_tensor[int((tr_frac+val_frac)*tr_len):])

        print('user {} train_len {} at_tr_len {} at_val_len {} at_te_len {}'.format(user_num,len(user_train_classifier_data_tensors[user_num]),len(train_classifier_data_tr_attack_tensors[user_num]),len(train_classifier_data_val_attack_tensors[user_num]),len(train_classifier_data_te_attack_tensors[user_num])))

        user_data=[user_train_classifier_data_tensors[user_num],
                   user_train_classifier_label_tensors[user_num],
                   train_classifier_data_tr_attack_tensors[user_num],
                   train_classifier_label_tr_attack_tensors[user_num],
                   train_classifier_data_val_attack_tensors[user_num],
                   train_classifier_label_val_attack_tensors[user_num],
                   train_classifier_data_te_attack_tensors[user_num],
                   train_classifier_label_te_attack_tensors[user_num]
                   ]

        pickle.dump(user_data, open(user_data_dir, 'wb'))
        user_tensor_lens.append(len(user_train_classifier_data_tensors[user_num]))

    # attack training/testing tensors - reference data
    user_ref_len=max(user_tensor_lens)
    user_ref_dataset=[ ref_dataset[0][:user_ref_len],ref_dataset[1][:user_ref_len] ]
    user_train_attack_data_tensor=torch.from_numpy(user_ref_dataset[0]).type(torch.FloatTensor)
    user_train_attack_label_tensor=torch.from_numpy(user_ref_dataset[1]).type(torch.LongTensor)
    
    pickle.dump([user_train_attack_data_tensor, user_train_attack_label_tensor], open(data_dir+'/user_attack_data.pkl', 'wb'))
    
    # central server's data tensor
    central_train_data=np.concatenate(user_train_data)
    central_train_label=np.concatenate(user_train_label)
    central_tr_len=len(central_train_data)
    print('Central train len: ',central_tr_len)
    
    central_train_classifier_data_tensor=torch.from_numpy(central_train_data).type(torch.FloatTensor)
    central_train_classifier_label_tensor=torch.from_numpy(central_train_label).type(torch.LongTensor)
    
    X=central_train_classifier_data_tensor
    Y=central_train_classifier_label_tensor
    
    tr_len=len(X)
    
    central_train_classifier_data_tr_attack_tensor=X[:int(tr_frac*tr_len)]
    central_train_classifier_label_tr_attack_tensor=Y[:int(tr_frac*tr_len)]
    central_train_classifier_data_val_attack_tensor=X[int(tr_frac*tr_len):int((tr_frac+val_frac)*tr_len)]
    central_train_classifier_label_val_attack_tensor=Y[int(tr_frac*tr_len):int((tr_frac+val_frac)*tr_len)]
    central_train_classifier_data_te_attack_tensor=X[int((tr_frac+val_frac)*tr_len):]
    central_train_classifier_label_te_attack_tensor=Y[int((tr_frac+val_frac)*tr_len):]

    if len(central_train_classifier_data_tensor) > len(ref_dataset[0]):
        ref_dataset[0]=np.concatenate([ref_dataset[0],pub_dataset[0]])
        ref_dataset[1]=np.concatenate([ref_dataset[1],pub_dataset[1]])
    else:
        ref_dataset[0]=ref_dataset[0][:len(central_train_classifier_data_tensor)]
        ref_dataset[1]=ref_dataset[1][:len(central_train_classifier_data_tensor)]
        
    central_train_attack_data_tensor=torch.from_numpy(ref_dataset[0]).type(torch.FloatTensor)
    central_train_attack_label_tensor=torch.from_numpy(ref_dataset[1]).type(torch.LongTensor)

    
    central_data=[central_train_classifier_data_tensor,
                  central_train_classifier_label_tensor,
                  central_train_classifier_data_tr_attack_tensor,
                  central_train_classifier_label_tr_attack_tensor,
                  central_train_classifier_data_val_attack_tensor,
                  central_train_classifier_label_val_attack_tensor,
                  central_train_classifier_data_te_attack_tensor,
                  central_train_classifier_label_te_attack_tensor,
                 ]
    pickle.dump(central_data, open(data_dir+'/central_data.pkl', 'wb'))
    print('central data saved')


    pickle.dump([central_train_attack_data_tensor, central_train_attack_label_tensor], open(data_dir+'/central_attack_data.pkl', 'wb'))

    # public data tensors
    public_data_tensor=torch.from_numpy(pub_dataset[0]).type(torch.FloatTensor)
    public_label_tensor=torch.from_numpy(pub_dataset[1]).type(torch.LongTensor)

    pickle.dump([public_data_tensor, public_label_tensor], open(data_dir+'/pub_data/pub_data.pkl', 'wb'))

    # test/validation data tensors
    test_data_tensor=torch.from_numpy(test_dataset[0]).type(torch.FloatTensor)
    test_label_tensor=torch.from_numpy(test_dataset[1]).type(torch.LongTensor)
    val_data_tensor=torch.from_numpy(val_dataset[0]).type(torch.FloatTensor)
    val_label_tensor=torch.from_numpy(val_dataset[1]).type(torch.LongTensor)
    
    pickle.dump([test_data_tensor,test_label_tensor,val_data_tensor,val_label_tensor], open(data_dir+'/global_data.pkl', 'wb'))

    if len(backdoor_datasets)>0:
        for p in range(len(backdoor_datasets)):
            at_tr_len=len(backdoor_datasets[p][0])
            a=np.arange(at_tr_len)

            np.random.shuffle(a)

            backdoor_data_tensors.append(torch.from_numpy(backdoor_datasets[p][0][a]).type(torch.FloatTensor))
            backdoor_label_tensors.append(torch.from_numpy(backdoor_datasets[p][1][a]).type(torch.LongTensor))

        at_te_len=len(backdoor_test_dataset[0])
        b=np.arange(at_te_len)
        np.random.shuffle(b)

        backdoor_test_data_tensor=torch.from_numpy(backdoor_test_dataset[0][b]).type(torch.FloatTensor)
        backdoor_test_label_tensor=torch.from_numpy(backdoor_test_dataset[1]).type(torch.LongTensor)

        print('''
        Backdoor attack training data len {}
        Backdoor attack test data len {}
        '''.format(len(backdoor_data_tensors[0]),
               len(backdoor_test_data_tensor)))

        pickle.dump([backdoor_data_tensors,backdoor_label_tensors,backdoor_test_data_tensor,backdoor_test_label_tensor], open(data_dir+'/backdoor_data.pkl', 'wb'))

    print('''User: ref_len {}
    Central server: tr_len {} ref_len {} at_tr_len {} at_val_len {} at_te_len {}
    Public Data len {}
    User test data len {}
    
    '''.format(len(user_train_attack_data_tensor),
               len(central_train_classifier_data_tensor),
               len(central_train_attack_data_tensor),
               len(central_train_classifier_data_tr_attack_tensor),
               len(central_train_classifier_data_val_attack_tensor),
               len(central_train_classifier_data_te_attack_tensor),
               len(public_data_tensor),
               len(test_data_tensor),
              ))
