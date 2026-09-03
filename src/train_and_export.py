import torch
import torch.nn.functional as F
f=open("matches_enriched_xg_top5_2014_2026.csv", "r")

Xlist=[]
Ylist=[]
Ylist1x2=[]
f.readline()
#27-39
#23-25
embList=[]
campionati={
    "E0": 0,
    "F1": 1,    
    "SP1": 2,
    "I1": 3,
    "D1": 4

}
risultati={
    "1":0,
    "X":1,
    "2":2
}



for riga in f:
    temp=[]
    temp2=[]
    parti=riga.split(',')
    embList.append(campionati[parti[0]])
    temp.append(float(parti[4]))
    temp.append(float(parti[5]))
    for i in range(11,66):
        temp.append(float(parti[i]))
    Xlist.append(temp)
    for j in range(9,11):
        temp2.append(float(parti[j]))
    Ylist.append(temp2)
    Ylist1x2.append(risultati[parti[8]])



epoche=10001
lunghadd=int(0.9*(len(Xlist)))
DIVtr = torch.tensor(embList[0:lunghadd], dtype=torch.long)
DIVtest = torch.tensor(embList[lunghadd:], dtype=torch.long)
Xtr=torch.tensor(Xlist[0:lunghadd], dtype=torch.float32)
Ytr=torch.tensor(Ylist[0:lunghadd], dtype=torch.float32)
Ytr1x2=torch.tensor(Ylist1x2[0:lunghadd], dtype=torch.long)
Xtest=torch.tensor(Xlist[lunghadd:], dtype=torch.float32)
Ytest=torch.tensor(Ylist[lunghadd:], dtype=torch.float32)
Ytest1x2=torch.tensor(Ylist1x2[lunghadd:], dtype=torch.long)
q25 = torch.quantile(Xtr, 0.25, dim=0)
q75 = torch.quantile(Xtr, 0.75, dim=0)
iqr = q75 - q25
median = torch.median(Xtr, dim=0).values

Xtr = (Xtr - median) / (iqr + 1e-8)
Xtest=(Xtest-median) / (iqr+ 1e-8)

emb=torch.empty(5,3)
torch.nn.init.xavier_normal_(emb)
w1=torch.empty(60,25)
torch.nn.init.xavier_normal_(w1)
b1=torch.zeros(25) 
w2=torch.empty(25,16)
torch.nn.init.xavier_normal_(w2)
b2=torch.zeros(16)
w3 = torch.empty(16, 2)
torch.nn.init.xavier_normal_(w3)
b3=torch.zeros(2)
w4=torch.empty(16, 3)
torch.nn.init.xavier_normal_(w4)
b4=torch.zeros(3)

parametri = [emb, w1,b1,w2,b2,w3,b3,w4,b4]

for p in parametri:
    p.requires_grad=True

lR=torch.nn.LeakyReLU(0.1)
dimBatch=2048
criterion = torch.nn.SmoothL1Loss(beta=1.0)

for i in range(epoche):
    ind = torch.randint(0, len(Xtr), (dimBatch,))
    emb_batch = emb[DIVtr[ind]]
    X_input = torch.cat([Xtr[ind], emb_batch], dim=1)
    X_input=F.batch_norm(X_input, running_mean=None, running_var=None, training=True)
    h=torch.tanh(X_input @ w1 + b1)
    h_drop=F.dropout(h, p=0.1, training=True)
    k=torch.tanh(h_drop @ w2 + b2)
    o=k @ w3 + b3
    outputxG=F.softplus(o)
    logits1x2= k @ w4 + b4
    loss_xg = criterion(outputxG, Ytr[ind])
    loss_1x2=F.cross_entropy(logits1x2, Ytr1x2[ind])
    loss = loss_xg + 0.3*loss_1x2


    if i<3000:
        rate=0.05
    elif i <=10000:
        rate=0.008
    else: rate=0.001
    for p in parametri:
        p.grad=None
    
    loss.backward()
    torch.nn.utils.clip_grad_norm_(parametri, max_norm=1.0)
    
    for p in parametri:
        p.data += -rate * p.grad
    #if i % 10000 == 0:
    print(f'{i}: {loss.data}')



with torch.no_grad():
    emb_test = emb[DIVtest]
    X_input = torch.cat([Xtest, emb_test], dim=1)
    X_input = F.batch_norm(X_input, running_mean=None, running_var=None, training=True) 
    h = torch.tanh(X_input @ w1 + b1)
    k = torch.tanh(h @ w2 + b2)
    o = k @ w3 + b3
    outputxG = F.softplus(o)
    logits1x2 = k @ w4 + b4

    loss_xG = criterion(outputxG, Ytest)
    loss_1x2 = F.cross_entropy(logits1x2, Ytest1x2)
    loss = loss_xG + 0.3 * loss_1x2
    mae = torch.abs(outputxG - Ytest).mean().item()
    predizioni_1x2 = torch.argmax(logits1x2, dim=1)
    accuratezza = (predizioni_1x2 == Ytest1x2).float().mean()

    print(f"Loss: {loss.item():.4f} | MAE: {mae:.3f} | Accuratezza 1X2: {accuratezza.item():.2%}")



checkpoint = {
    "emb": emb.detach(),
    "w1": w1.detach(),
    "b1": b1.detach(),
    "w2": w2.detach(),
    "b2": b2.detach(),
    "w3": w3.detach(),
    "b3": b3.detach(),
    "w4": w4.detach(),
    "b4": b4.detach(),
    "median": median.detach(),
    "iqr": iqr.detach(),
    "campionati": campionati,
}

torch.save(checkpoint, "modello_calcio_v1.pt")


















