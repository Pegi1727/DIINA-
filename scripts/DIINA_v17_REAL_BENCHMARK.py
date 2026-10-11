# ============================================================
# DIINA v17 REAL EXPERIMENTAL BENCHMARK
# Transformer vs DIINA on SCAN-Like Sequence Task
# ============================================================

import torch
import torch.nn as nn
import torch.optim as optim
import random
import time
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Seed
# -----------------------------

torch.manual_seed(42)
random.seed(42)


device = "cuda" if torch.cuda.is_available() else "cpu"



# ============================================================
# Synthetic SCAN-Lite Dataset
# ============================================================

commands = [
    "jump",
    "walk",
    "run",
    "look",
    "turn"
]


actions = [
    "JUMP",
    "WALK",
    "RUN",
    "LOOK",
    "TURN"
]


pairs=[]


for c,a in zip(commands,actions):

    for rep in range(1,6):

        inp = [c]*rep
        out = [a]*rep

        pairs.append(
            (inp,out)
        )


random.shuffle(pairs)



# vocabulary

src_vocab = list(set(
    [x for p in pairs for x in p[0]]
))

tgt_vocab = list(set(
    [x for p in pairs for x in p[1]]
))


src2id={
    w:i for i,w in enumerate(src_vocab)
}

tgt2id={
    w:i for i,w in enumerate(tgt_vocab)
}



def encode(x,v):

    return [
        v[w]
        for w in x
    ]



data=[]

for x,y in pairs:

    data.append(
        (
            torch.tensor(
                encode(x,src2id)
            ),
            torch.tensor(
                encode(y,tgt2id)
            )
        )
    )



split=int(len(data)*0.8)

train=data[:split]
test=data[split:]



# ============================================================
# Models
# ============================================================


class TransformerBaseline(nn.Module):

    def __init__(self):

        super().__init__()

        self.emb=nn.Embedding(
            len(src_vocab),
            32
        )

        encoder_layer=nn.TransformerEncoderLayer(
            d_model=32,
            nhead=4,
            batch_first=True
        )

        self.encoder=nn.TransformerEncoder(
            encoder_layer,
            2
        )

        self.out=nn.Linear(
            32,
            len(tgt_vocab)
        )


    def forward(self,x):

        x=self.emb(x)

        x=self.encoder(x)

        return self.out(x)



# ============================================================
# DIINA
# ============================================================


class DIINA(nn.Module):

    def __init__(self):

        super().__init__()

        self.emb=nn.Embedding(
            len(src_vocab),
            32
        )

        self.gate=nn.Sequential(

            nn.Linear(32,32),
            nn.Sigmoid()

        )


        self.state=nn.GRU(
            32,
            32,
            batch_first=True
        )


        self.out=nn.Linear(
            32,
            len(tgt_vocab)
        )


    def forward(self,x):

        x=self.emb(x)

        h,_=self.state(x)

        inhibition=self.gate(h)

        h=h*(1-inhibition)

        return self.out(h)



# ============================================================
# Training
# ============================================================


def train_model(model,name):

    model.to(device)

    opt=optim.Adam(
        model.parameters(),
        lr=0.003
    )


    loss_fn=nn.CrossEntropyLoss()


    start=time.time()


    for epoch in range(80):

        total=0


        random.shuffle(train)


        for x,y in train:

            x=x.unsqueeze(0).to(device)
            y=y.to(device)


            pred=model(x)


            loss=loss_fn(
                pred.squeeze(0),
                y
            )


            opt.zero_grad()

            loss.backward()

            opt.step()


            total+=loss.item()



        if epoch%20==0:

            print(
                name,
                "epoch",
                epoch,
                "loss",
                round(total,3)
            )


    return time.time()-start




# ============================================================
# Evaluation
# ============================================================


def evaluate(model,name):

    model.eval()

    correct=0
    total=0


    with torch.no_grad():

        for x,y in test:


            x=x.unsqueeze(0).to(device)


            pred=model(x)


            pred=torch.argmax(
                pred,
                dim=-1
            ).cpu().squeeze()


            if torch.equal(
                pred,
                y
            ):

                correct+=1


            total+=1


    acc=correct/total


    print(
        name,
        "Accuracy:",
        round(acc,4)
    )


    return acc




# ============================================================
# Run
# ============================================================


models={

"Transformer":
TransformerBaseline(),

"DIINA":
DIINA()

}



results=[]


for name,model in models.items():

    print("\nTraining",name)

    t=train_model(
        model,
        name
    )

    acc=evaluate(
        model,
        name
    )


    results.append({

        "Model":name,
        "Accuracy":acc,
        "Training_Time":t

    })



# ============================================================
# Save
# ============================================================


df=pd.DataFrame(results)


print("\n========== FINAL RESULTS ==========")

print(df)



df.to_csv(
"DIINA_v17_REAL_RESULTS.csv",
index=False
)



plt.figure(figsize=(6,4))


plt.bar(
df.Model,
df.Accuracy
)


plt.ylabel(
"Accuracy"
)

plt.title(
"DIINA v17 Real Benchmark"
)


plt.ylim(
0,
1
)


plt.savefig(
"DIINA_v17_Result.png",
dpi=300
)


print(
"""
============================

FILES:

DIINA_v17_REAL_RESULTS.csv

DIINA_v17_Result.png

============================
"""
)