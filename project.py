'''
Preferible llegir l'article abans que l'script

Index:
    1) Llibries
    2) Data Loading
    3) Classical Transfer Learning
    4) Quantum Transfer Learning
        4.1) Quantum layers
        4.2) Quantum circuit
        4.3) Dressed quantum circuit
        4.4) Hybrid Classical-QUanutm network
    5) Training and Evaluation
    6) Prediciton (test models)

Recomanació per provar el programa: posar pocs epochs (entre 5-15 epochs)

Tarda entre 20min i 2 hores en fer tots el epochs d'entrenament a la CPU
'''




# Llibreries
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt

import pennylane as qml


import time

# Escull la GPU si es pot, sinó la CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


'''

DATA LOADING

Aquí carrego les imatges, tenint en compte que tinc 5 tipus d imatges: zebres, pinguins, gossos, girafes i aranyes.
I que una part de les imatges serviran per entrenar les xarxes neuronals, unes altres per fer validacions i les últimes
per fer tests posteriors.

'''

# Transformacions que aplicaré a les imatges que importi
# Les imatges són de diferents mides, per tant cal que siguin normalitzades
# per aixó aplico aquestes transformacions (transformacions standards utilitzades a pytorch)
image_transforms = {
    'train': transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
}



# Carrego les imatges
 
# els directoris de les carpetes
train_directory = 'train'
valid_directory = 'valid'
test_directory = 'test'
 
# Batch size
bs = 4
 
# Nombre de classes diferents d'animals que el classificador podrà diferenciar
num_classes = 5
 
# Carreguem la data utilitzant Image folder
data = {
    'train': datasets.ImageFolder(root=train_directory, transform=image_transforms['train']),
    'valid': datasets.ImageFolder(root=valid_directory, transform=image_transforms['valid']),
    'test': datasets.ImageFolder(root=test_directory, transform=image_transforms['test'])
}
 
# Mida de la data, serà utilitzat per calcular l'average loss i accuracy posteriorment
train_data_size = len(data['train'])
valid_data_size = len(data['valid'])
test_data_size = len(data['test'])
 
# El dataloader ens permet iterar per totes les imatges
train_data = torch.utils.data.DataLoader(data['train'], batch_size=bs, shuffle=True)
valid_data = torch.utils.data.DataLoader(data['valid'], batch_size=bs, shuffle=True)
test_data = torch.utils.data.DataLoader(data['test'], batch_size=bs, shuffle=True)
 
#labels
class_names = data['train'].classes



'''

CLASSIC TRANSFER LEARNING

'''

# Carrego la xarxa pre entrenada ResNet18
xarxa = models.resnet18(pretrained=True)

#Fixem els paràmetres de la xarxa preentrenada
for param in xarxa.parameters():
    param.requires_grad = False
    

# Canviem la última capa de la xarxa pre entrenada ResNet18 per aplicar el Transfer Learning
fc_inputs = xarxa.fc.in_features

#Seqüència de capes 
xarxa.fc = nn.Sequential(
    nn.Linear(fc_inputs, 512),
    nn.ReLU(),
    nn.Linear(512, 5),
    nn.LogSoftmax(dim=1))


#la funció device ja analitza si es pot fer servir la GPU sinó es fa servir la CPU
xarxa = xarxa.to(device)



'''

QUANTUM TRANSFER LEARNING

'''

#Paràmatres que utilitzaré per construir el circuit quàntic i el dresssed quantum circuit
n_qubits = 4                # Nombre de qubits
q_depth = 8                 # Nombre de capes variacionals
q_delta = 0.01              # Paràmetre arbitrari
dev = qml.device("default.qubit", wires=n_qubits)   #simulador

#Quantum layers que composaràn el circuit quàntic
def H_layer(nqubits):
    """capa amb el gate hadamard
    """
    for idx in range(nqubits):
        qml.Hadamard(wires=idx)


def RY_layer(w):
    """capa que aplica una rotació en l'eix y
    """
    for idx, element in enumerate(w):
        qml.RY(element, wires=idx)


def entangling_layer(nqubits):
    """capa de gates CNOT per entrellaçar els qubits
    """
    # S'aplicarà una cosa semblant :
    # CNOT  CNOT  CNOT  CNOT...  CNOT
    #   CNOT  CNOT  CNOT...  CNOT
    for i in range(0, nqubits - 1, 2):  # loop sobre els qubits parells: i=0,2,...N-2
        qml.CNOT(wires=[i, i + 1])
    for i in range(1, nqubits - 1, 2):  # Loop sobre els qubits inaprells:  i=1,3,...N-3
        qml.CNOT(wires=[i, i + 1])
    
    


#Circuit quàntic
@qml.qnode(dev, interface="torch")  #simulador
def quantum_net(q_input_features, q_weights_flat):
    """
    circuit variacional quàntic
    """

    # Canviem el format per poder treballar amb els pesos
    q_weights = q_weights_flat.reshape(q_depth, n_qubits)

    # Embedding Layer
    H_layer(n_qubits)
    RY_layer(q_input_features)

    # Circuit variacional
    for k in range(q_depth):
        entangling_layer(n_qubits)
        RY_layer(q_weights[k])

    # Valors esperats en la base Z
    exp_vals = [qml.expval(qml.PauliZ(position)) for position in range(n_qubits)]
    return tuple(exp_vals)


#DRESSED QUANTUM CIRCUIT
class DressedQuantumNet(nn.Module):
    def __init__(self):
        """
        Definim la capa dressed quantum
        """

        super().__init__()
        self.pre_net = nn.Linear(512, n_qubits)
        self.q_params = nn.Parameter(q_delta * torch.randn(q_depth * n_qubits))
        self.post_net = nn.Linear(n_qubits, 5)

    def forward(self, input_features):
        """
        Definim el moviment de la informació i el seu tractament al llarg de la dressed quantum layer
        """
        # Obtenim els inputs pel circuit quàntic
        # reduïnt les dimensions de 512 a 4
        pre_out = self.pre_net(input_features)
        q_in = torch.tanh(pre_out) * np.pi / 2.0    #equivalent al ReLu
        
        # Apliquem el circuit quàntic a cada input i guardem els outputs a q_out
        q_out = torch.Tensor(0, n_qubits)
        q_out = q_out.to(device)
        for elem in q_in:
            q_out_elem = quantum_net(elem, self.q_params).float().unsqueeze(0)
            q_out = torch.cat((q_out, q_out_elem))
            
        # retorna la predicció després de tornar a convertir els outputs
        # que estaven en format d'estat quàntic a vectors reals
        return self.post_net(q_out)



#HYBRID CLASSICAL-QUANTUM NETWORK

# importo la pretrained ResNet18 Model
xarxa_quantum = models.resnet18(pretrained=True)

#Fixem els paràmetres de la xarxa preentrenada
for param in xarxa_quantum.parameters():
    param.requires_grad = False
    
#Canviem la última capa de la resnet18 pel dressed quantum circuit
xarxa_quantum.fc = DressedQuantumNet()

# Utilizar CPU o GPU segons la variable device definida anteriorment
xarxa_quantum = xarxa_quantum.to(device)




'''

TRAINING AND EVALUATION

'''

# Defineixo el learning rate, la loss function i l'optimizer
step = 0.0003   #learning rate
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(xarxa.parameters(), lr = step)
optimizer_quantum = optim.Adam(xarxa_quantum.parameters(), lr = step)

epochs = 5         #Nombre d'iteracions a tota la data per entrenar les xarxes

#Guardaré els valors de Loss i accuracy a cada epoch per cada xarxa
t_loss =[ ]
t_acc = []
v_loss = []
v_acc = []
t_loss_q = []
t_acc_q = []
v_loss_q = []
v_acc_q = []

#Training
for epoch in range(epochs):
    epoch_start = time.time()
    print("Epoch: {}/{}".format(epoch+1, epochs))
     
    # Posem en mode entrenament
    xarxa.train()
     
    # Inicialitzo la Loss i la Accuracy xarxa Clàssica
    train_loss = 0.0
    train_acc = 0.0
     
    valid_loss = 0.0
    valid_acc = 0.0
 
    for i, (inputs, labels) in enumerate(train_data):
        
        inputs = inputs.to(device)
        labels = labels.to(device)
         
        # Reiniciem els gradients existents
        optimizer.zero_grad()
         
        # Apliquem la xarxa als inputs
        outputs = xarxa(inputs)
         
        # Calculem la loss
        loss = loss_func(outputs, labels)
         
        # propaguem els gradients per la xarxa
        loss.backward()
         
        # Optimitzem els valors
        optimizer.step()
        
        #Calculo el loss total per cada batch i l'afegeixo al train_loss
        train_loss += loss.item() * inputs.size(0)
         
        # Calculo l'accuracy
        ret, predictions = torch.max(outputs.data, 1)
        correct_counts = predictions.eq(labels.data.view_as(predictions))
         
        # Calculo la mitjana dels valors de l'accurracy
        acc = torch.mean(correct_counts.type(torch.FloatTensor))
         
        # Calculo l'accuracy total per cada batch i l'afegeixo al train_acc
        train_acc += acc.item() * inputs.size(0)
         
                
        
    # Inicialitzo els valors del loss i l'accuracy per la xarxa quàntica
    train_loss_quantum = 0.0
    train_acc_quantum = 0.0
     
    valid_loss_quantum = 0.0
    valid_acc_quantum = 0.0
 
    for k, (inputs, labels) in enumerate(train_data):
 
        inputs = inputs.to(device)
        labels = labels.to(device)
         
        # Reiniciem els gradients existents
        optimizer_quantum.zero_grad()
         
        # Apliquem la xarxa als inputs
        outputs_quantum = xarxa_quantum(inputs)
         
        # Calculem la loss
        loss_quantum = loss_func(outputs_quantum, labels)
         
        # propaguem els gradients per la xarxa
        loss_quantum.backward()
         
        # Optimitzem els valors
        optimizer_quantum.step()
         
        #Calculo el loss total per cada batch i l'afegeixo al train_loss_quantum
        train_loss_quantum += loss_quantum.item() * inputs.size(0)
         
        # Calculo l'accuracy
        ret_quantum, predictions_quantum = torch.max(outputs_quantum.data, 1)
        correct_counts_quantum = predictions_quantum.eq(labels.data.view_as(predictions_quantum))
         
        # Calculo la mitjana dels valors de l'accurracy
        acc_quantum = torch.mean(correct_counts_quantum.type(torch.FloatTensor))
         
        # Calculo l'accuracy total per cada batch i l'afegeixo al train_acc_quantum
        train_acc_quantum += acc_quantum.item() * inputs.size(0)
         
        

    # Validació (no modifiquem el gradient)
    with torch.no_grad():
     
        # Posem en evaluation mode la xarxa Clàssica
        xarxa.eval()
     
        # Validation loop
        for j, (inputs, labels) in enumerate(valid_data):
            inputs = inputs.to(device)
            labels = labels.to(device)
     
           
            outputs = xarxa(inputs)
            loss = loss_func(outputs, labels)
     
            #Calculo el loss total per cada batch i l'afegeixo al valid_loss
            valid_loss += loss.item() * inputs.size(0)
     
            # Calculo la validation accuracy
            ret, predictions = torch.max(outputs.data, 1)
            correct_counts = predictions.eq(labels.data.view_as(predictions))
     
            # Calculo la mitjana de l'accuracy
            acc = torch.mean(correct_counts.type(torch.FloatTensor))
     
            # Calculo l'accuracy total per cada batch i l'afegeixo al valid_acc
            valid_acc += acc.item() * inputs.size(0)
     
        
     
        # Posem a evaluation mode la xarxa Quàntica
        xarxa_quantum.eval()
     
        # Loop de validació
        for l, (inputs, labels) in enumerate(valid_data):
            inputs = inputs.to(device)
            labels = labels.to(device)
     
            
            outputs_quantum = xarxa_quantum(inputs)
            loss = loss_func(outputs_quantum, labels)
            
            # Calculem la loss total per el batch i l'afegim al valid_loss_quantum
            valid_loss_quantum += loss.item() * inputs.size(0)
     
            # Calculem la validation accuracy
            ret, predictions = torch.max(outputs_quantum.data, 1)
            correct_counts = predictions.eq(labels.data.view_as(predictions))
            
            # Convertim la correct_counts a float i calculem la mitjana
            acc = torch.mean(correct_counts.type(torch.FloatTensor))
     
            # Calculem la accuracy toatl pel batch i l'afegim al valid_acc_quantum
            valid_acc_quantum += acc.item() * inputs.size(0)
     
            
     
    # Trobo l'average training loss i training accuracy per cada xarxa
    avg_train_loss = train_loss/train_data_size
    avg_train_acc = train_acc/float(train_data_size)
    avg_train_loss_quantum = train_loss_quantum/train_data_size
    avg_train_acc_quantum = train_acc_quantum/float(train_data_size)
     
    # Trobo l'average validation loss i validation accuracy per cada xarxa
    avg_valid_loss = valid_loss/valid_data_size
    avg_valid_acc = valid_acc/float(valid_data_size)
    avg_valid_loss_quantum = valid_loss_quantum/valid_data_size
    avg_valid_acc_quantum = valid_acc_quantum/float(valid_data_size)
    
    #Els fico en una llista per poder graficarlos posteriorment
    t_loss.append(avg_train_loss)   
    v_loss.append(avg_valid_loss)
    t_acc.append(avg_train_acc)   
    v_acc.append(avg_valid_acc)
    t_loss_q.append(avg_train_loss_quantum)   
    v_loss_q.append(avg_valid_loss_quantum)
    t_acc_q.append(avg_train_acc_quantum)   
    v_acc_q.append(avg_valid_acc_quantum)
      
    #Serveix per indicar al contador que pari           
    epoch_end = time.time()
     
    print("Classical Net:\ Epoch : {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}%, \nValidation : Loss : {:.4f}, Accuracy: {:.4f}%, Time: {:.4f}s".format(epoch, avg_train_loss, avg_train_acc*100, avg_valid_loss, avg_valid_acc*100, epoch_end-epoch_start))
    print("Quantum Net:\ Epoch : {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}%, \nValidation : Loss : {:.4f}, Accuracy: {:.4f}%, Time: {:.4f}s".format(epoch, avg_train_loss_quantum, avg_train_acc_quantum*100, avg_valid_loss_quantum, avg_valid_acc_quantum*100, epoch_end-epoch_start))


#Grafico els resultats, comparant les dues xarxes
plt.figure('Comparació Xarxes Neuronals')
plt.subplot(221)
plt.title('Train Loss: Quantum vs Classical')
plt.xlabel('Epochs')
plt.ylabel('Train Loss')
plt.plot(np.arange(0,epochs,1), t_loss, label='Classical')
plt.plot(np.arange(0,epochs,1), t_loss_q, label='Quantum')
plt.legend(loc="upper right")
plt.subplot(222)
plt.title('Validation Loss: Quantum vs Classical')
plt.xlabel('Epochs')
plt.ylabel('Validation Loss')
plt.plot(np.arange(0,epochs,1), v_loss, label='Classical')
plt.plot(np.arange(0,epochs,1), v_loss_q, label='Quantum')
plt.legend(loc="upper right")
plt.subplot(223)
plt.title('Train Accuracy: Quantum vs Classical')
plt.xlabel('Epochs')
plt.ylabel('Train Accuracy')
plt.plot(np.arange(0,epochs,1), t_acc, label='Classical')
plt.plot(np.arange(0,epochs,1), t_acc_q, label='Quantum')
plt.legend(loc="lower right")
plt.subplot(224)
plt.title('Validation Accuracy: Quantum vs Classical')
plt.xlabel('Epochs')
plt.ylabel('Validation Accuracy')
plt.plot(np.arange(0,epochs,1), t_acc, label='Classical')
plt.plot(np.arange(0,epochs,1), t_acc_q, label='Quantum')
plt.legend(loc="lower right")
plt.show()


'''

PREDICTION

Defineixo un imshow modificat per poder mostrar les imatges test i la funció predict
que serveix per provar els models i veure si encerta amb la classifiació d'animals

'''

def imshow(inp, title=None):
    """Imshow per Tensors"""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)

    

def predict(model, num_images=4):
    
    #Posem em modo eval
    model.eval()
    images_so_far = 0       #variable de control per les imatges que s'han de ensenyar
    
    
    if model == xarxa:
        fig = plt.figure('Classical')
    elif model == xarxa_quantum:
        fig = plt.figure('Quantum')

    with torch.no_grad():
        #Agafem imatges del grup d'imatges test
        for i, (inputs, labels) in enumerate(test_data):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            #Apliquem el nostre model als inputs
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)    #retorna el valor maxim dels outputs


            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicció: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    return
        

# Predicció de cada xarxa a 4 imatges random
predict(xarxa_quantum)
predict(xarxa)

