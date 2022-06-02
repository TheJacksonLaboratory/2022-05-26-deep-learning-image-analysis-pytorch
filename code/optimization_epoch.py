trn_loss = []
for i, (x, t) in enumerate(cifar_loader):
    optimizer.zero_grad()
    
    x = x.cuda()
    t = t.cuda()
    
    y = net(x)
    
    loss = criterion(y, t)
    
    loss.backward()
    
    trn_loss.append(loss.item())
    
    optimizer.step()
    
    if i % 100 == 0:
        print('[%i] Loss %.8f' % (i, loss))
