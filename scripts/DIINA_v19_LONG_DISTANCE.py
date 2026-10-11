# ============================================================
# DIINA v19 LONG DISTANCE DEPENDENCY BENCHMARK
# Associative Recall Task
# Transformer vs Mamba vs DIINA
# ============================================================

import torch
import torch.nn as nn
import torch.optim as optim

import random
import time
import pandas as pd
import matplotlib.pyplot as plt


print("Torch:", torch.__version__)


device = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", device)


torch.manual_seed(42)
random.seed(42)



# ============================================================
# Vocabulary
# ============================================================

KEYS = [
    "A",
    "B",
    "C",
    "D"
]


NOISE = [
    "x",
    "y",
    "z",
    "n"
]


VOCAB = KEYS + NOISE


stoi={
    w:i for i,w in enumerate(VOCAB)
}


# output classes

key_id={
    k:i for i,k in enumerate(KEYS)
}




# ============================================================
# Dataset
# ============================================================


def generate_dataset(distance, samples=500):

    data=[]


    for _ in range(samples):


        key=random.choice(KEYS)


        distractors=[
            random.choice(NOISE)
            for _ in range(distance)
        ]


        # key at beginning
        sequence=[
            key
        ] + distractors + [

            "QUERY"

        ]



        target=key_id[key]


        data.append(
            (
                sequence,
                target
            )
        )


    return data




stoi["QUERY"]=len(stoi)



def encode(seq):

    return torch.tensor(
        [
            stoi[x]
            for x in seq
        ],
        dtype=torch.long
    )



# ============================================================
# Transformer
# ============================================================


class TransformerModel(nn.Module):

    def __init__(self):

        super().__init__()


        self.emb=nn.Embedding(
            len(stoi),
            64
        )


        layer=nn.TransformerEncoderLayer(
            d_model=64,
            nhead=4,
            batch_first=True
        )


        self.encoder=nn.TransformerEncoder(
            layer,
            3
        )


        self.fc=nn.Linear(
            64,
            4
        )



    def forward(self,x):

        h=self.emb(x)

        h=self.encoder(h)


        # use QUERY position

        q=h[:,-1,:]


        return self.fc(q)




# ============================================================
# Mamba-like baseline
# ============================================================


class MambaBaseline(nn.Module):

    def __init__(self):

        super().__init__()


        self.emb=nn.Embedding(
            len(stoi),
            64
        )


        self.rnn=nn.GRU(
            64,
            64,
            batch_first=True
        )


        self.fc=nn.Linear(
            64,
            4
        )


    def forward(self,x):

        h,_=self.rnn(
            self.emb(x)
        )


        return self.fc(
            h[:,-1,:]
        )





# ============================================================
# DIINA
# ============================================================


class DIINA(nn.Module):

    def __init__(self):

        super().__init__()


        self.emb=nn.Embedding(
            len(stoi),
            64
        )


        self.rnn=nn.GRU(
            64,
            64,
            batch_first=True
        )


        self.inhibition=nn.Sequential(

            nn.Linear(64,64),
            nn.Sigmoid()

        )


        self.fc=nn.Linear(
            64,
            4
        )



    def forward(self,x):


        h,_=self.rnn(
            self.emb(x)
        )


        gate=self.inhibition(h)


        # dynamic suppression of irrelevant states

        h=h*(1-gate)



        return self.fc(
            h[:,-1,:]
        )




# ============================================================
# Training
# ============================================================


def train(model,data,epochs=40):


    model.to(device)


    optimizer=optim.Adam(
        model.parameters(),
        lr=0.001
    )


    loss_fn=nn.CrossEntropyLoss()


    start=time.time()



    for epoch in range(epochs):


        random.shuffle(data)


        for seq,target in data:


            x=encode(seq).unsqueeze(0).to(device)


            y=torch.tensor(
                [target]
            ).to(device)



            pred=model(x)


            loss=loss_fn(
                pred,
                y
            )


            optimizer.zero_grad()

            loss.backward()

            optimizer.step()



    return time.time()-start




# ============================================================
# Evaluation
# ============================================================


def evaluate(model,data):


    model.eval()


    correct=0


    with torch.no_grad():

        for seq,target in data:


            x=encode(seq).unsqueeze(0).to(device)


            pred=torch.argmax(
                model(x)
            ).item()


            if pred==target:

                correct+=1



    return correct/len(data)




# ============================================================
# Experiment
# ============================================================


results=[]


distances=[

50,
100,
200,
500

]



models={

"Transformer":TransformerModel,

"Mamba":MambaBaseline,

"DIINA":DIINA

}




for d in distances:


    print("\nDependency length:",d)


    train_data=generate_dataset(
        d,
        800
    )


    test_data=generate_dataset(
        d,
        300
    )


    for name,cls in models.items():


        model=cls()


        t=train(
            model,
            train_data
        )


        acc=evaluate(
            model,
            test_data
        )


        print(
            name,
            "Accuracy:",
            round(acc,4),
            "Time:",
            round(t,2)
        )


        results.append({

            "Model":name,

            "Distance":d,

            "Accuracy":acc,

            "Time":t

        })




# ============================================================
# Save
# ============================================================


df=pd.DataFrame(results)


print("\n========== RESULTS ==========")

print(df)



df.to_csv(
"DIINA_v19_LongDistance.csv",
index=False
)



plt.figure(
figsize=(8,5)
)



for m in df.Model.unique():

    temp=df[
        df.Model==m
    ]


    plt.plot(
        temp.Distance,
        temp.Accuracy,
        marker="o",
        label=m
    )



plt.xlabel(
"Dependency Distance"
)


plt.ylabel(
"Accuracy"
)


plt.title(
"DIINA v19 Long Distance Dependency"
)


plt.legend()


plt.savefig(
"DIINA_v19_LongDistance.png",
dpi=300
)


print("""
FILES:

DIINA_v19_LongDistance.csv

DIINA_v19_LongDistance.png

""")