# ============================================================
# DIINA v18 LONG CONTEXT STRESS BENCHMARK
# Transformer vs DIINA
# Long Sequence + Noise Robustness
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



# -----------------------------
# Seed
# -----------------------------

torch.manual_seed(42)
random.seed(42)



# ============================================================
# Dataset Generator
# ============================================================


VOCAB = [
    "jump",
    "walk",
    "run",
    "look",
    "turn"
]


LABELS = {
    "jump":0,
    "walk":1,
    "run":2,
    "look":3,
    "turn":4
}



def generate_dataset(length, noise):

    data=[]


    for _ in range(200):

        token=random.choice(VOCAB)

        seq=[token]*length


        noisy_seq=[]


        for t in seq:

            if random.random()<noise:

                noisy_seq.append(
                    random.choice(VOCAB)
                )

            else:

                noisy_seq.append(t)


        target=[
            LABELS[token]
            for _ in range(length)
        ]


        data.append(
            (
                noisy_seq,
                torch.tensor(target)
            )
        )


    return data




def encode(x):

    return torch.tensor(
        [LABELS[t] for t in x],
        dtype=torch.long
    )




# ============================================================
# Transformer
# ============================================================


class TransformerModel(nn.Module):


    def __init__(self):

        super().__init__()


        self.emb=nn.Embedding(
            5,
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
            5
        )


    def forward(self,x):

        x=self.emb(x)

        x=self.encoder(x)

        return self.fc(x)



# ============================================================
# DIINA
# ============================================================


class DIINA(nn.Module):


    def __init__(self):

        super().__init__()


        self.emb=nn.Embedding(
            5,
            64
        )


        self.rnn=nn.GRU(
            64,
            64,
            batch_first=True
        )


        self.inhibitory_gate=nn.Sequential(

            nn.Linear(64,64),
            nn.Sigmoid()

        )


        self.fc=nn.Linear(
            64,
            5
        )


    def forward(self,x):


        x=self.emb(x)


        h,_=self.rnn(x)


        gate=self.inhibitory_gate(h)


        h=h*(1-gate)


        return self.fc(h)




# ============================================================
# Train
# ============================================================


def train(model,data):


    model.to(device)


    optimizer=optim.Adam(
        model.parameters(),
        lr=0.001
    )


    loss_fn=nn.CrossEntropyLoss()


    start=time.time()


    model.train()


    for epoch in range(30):


        random.shuffle(data)


        for x,y in data:


            x=encode(x).unsqueeze(0).to(device)

            y=y.to(device)



            pred=model(x)



            loss=loss_fn(
                pred.squeeze(0),
                y
            )



            optimizer.zero_grad()

            loss.backward()

            optimizer.step()



    return time.time()-start




# ============================================================
# Evaluate
# ============================================================


def evaluate(model,data):


    model.eval()


    correct=0

    total=0


    with torch.no_grad():


        for x,y in data:


            x=encode(x).unsqueeze(0).to(device)


            pred=model(x)


            pred=torch.argmax(
                pred,
                dim=-1
            ).cpu().squeeze()



            correct += (
                pred==y
            ).sum().item()


            total+=len(y)



    return correct/total




# ============================================================
# Experiment
# ============================================================


results=[]


lengths=[
    10,
    50,
    100,
    200
]


noises=[
    0.0,
    0.1,
    0.2
]



for length in lengths:


    for noise in noises:


        print(
            "\nLength:",
            length,
            "Noise:",
            noise
        )


        train_data=generate_dataset(
            length,
            noise
        )


        test_data=generate_dataset(
            length,
            noise
        )


        for name,model in [

            ("Transformer",
             TransformerModel()),

            ("DIINA",
             DIINA())

        ]:


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
                "acc:",
                round(acc,4),
                "time:",
                round(t,2)
            )


            results.append({

                "Model":name,
                "Length":length,
                "Noise":noise,
                "Accuracy":acc,
                "Time":t

            })




# ============================================================
# Save
# ============================================================


df=pd.DataFrame(results)


print("\n========== FINAL RESULTS ==========")

print(df)



df.to_csv(
"DIINA_v18_Long_Context_Results.csv",
index=False
)



summary=df.groupby(
"Model"
)[
"Accuracy"
].mean()



print("\nMODEL SUMMARY")

print(summary)



plt.figure(
figsize=(8,5)
)


for model in df.Model.unique():

    temp=df[
        df.Model==model
    ].groupby(
        "Length"
    )[
        "Accuracy"
    ].mean()


    plt.plot(
        temp.index,
        temp.values,
        marker="o",
        label=model
    )


plt.xlabel(
"Sequence Length"
)

plt.ylabel(
"Accuracy"
)

plt.title(
"DIINA v18 Long Context Stress Test"
)


plt.legend()


plt.savefig(
"DIINA_v18_Long_Context_Result.png",
dpi=300
)


print("""
FILES GENERATED:

DIINA_v18_Long_Context_Results.csv

DIINA_v18_Long_Context_Result.png

""")